import os
import json
import numpy as np
import pandas as pd
import faiss
import gradio as gr
import google.genai as genai

from datetime import datetime, date
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

client_gemini = genai.Client(api_key=GEMINI_API_KEY)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

DATA_DIR = Path("energy_planner_data")
DATA_DIR.mkdir(exist_ok=True)
LOG_FILE = DATA_DIR / "activity_log.json"
PLAN_FILE = DATA_DIR / "plans_log.json"

ACTIVITY_CATEGORIES = [
    "Social interaction",
    "Work / Study",
    "Sensory demand",
    "Physical activity",
    "Admin / Errands",
    "Self-care / Rest",
    "Travel / Commute",
    "Other"
]

SYSTEM_PROMPT = """
You are EnergyPlanner, a calm and practical daily planning assistant designed specifically
for autistic and sensory-sensitive people.

Your role is to help the user plan their next day in a way that protects their energy,
avoids overload, and builds in genuine recovery time.

Your communication style:
- Plain, direct language. No vague phrases like "try to relax" or "listen to your body".
- Short sentences. One idea at a time.
- Never condescending. Never over-cheerful.
- If something looks risky, say so plainly.
- Always suggest a recovery buffer after any activity rated 7+ energy cost.
- Never suggest more than 3 high-demand activities in a single day plan.

Output format:
1. A brief honest assessment of how the planned day looks (2-3 sentences max).
2. The day plan with suggested time slots and recovery buffers clearly marked.
3. One specific thing to protect.
4. One optional thing to move if energy is low on the day.

Keep the total response under 350 words.
"""

# ── Data helpers ───────────────────────────────────────────────────────────────

def load_logs():
    if LOG_FILE.exists():
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return []

def save_logs(logs):
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

def append_activity(activity, energy_cost, category, notes=""):
    logs = load_logs()
    entry = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "date": str(date.today()),
        "timestamp": datetime.now().isoformat(),
        "activity": activity.strip(),
        "energy_cost": energy_cost,
        "category": category,
        "notes": notes.strip()
    }
    logs.append(entry)
    save_logs(logs)
    return entry

def get_energy_stats(logs):
    if not logs:
        return {}
    df = pd.DataFrame(logs)
    return {
        "total_entries": len(df),
        "avg_energy_cost": round(df["energy_cost"].mean(), 1),
        "most_draining_category": df.groupby("category")["energy_cost"].mean().idxmax(),
        "most_draining_activity": df.loc[df["energy_cost"].idxmax(), "activity"],
        "days_tracked": df["date"].nunique()
    }

# ── RAG pipeline ───────────────────────────────────────────────────────────────

def build_vector_store(logs):
    if not logs:
        return None, []
    documents = [
        f"Date: {l['date']} | Activity: {l['activity']} | "
        f"Category: {l['category']} | Energy cost: {l['energy_cost']}/10"
        + (f" | Notes: {l['notes']}" if l.get("notes") else "")
        for l in logs
    ]
    embeddings = embed_model.encode(documents, convert_to_numpy=True).astype(np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, documents

def retrieve_similar(query, index, documents, top_k=5):
    if index is None or not documents:
        return []
    q = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(q, min(top_k, len(documents)))
    return [{"document": documents[i], "distance": float(distances[0][n])}
            for n, i in enumerate(indices[0]) if i < len(documents)]

# ── Plan generation ────────────────────────────────────────────────────────────

def generate_plan(scheduled_activities, extra_notes=""):
    logs = load_logs()
    index, documents = build_vector_store(logs)
    query = " ".join([a["activity"] for a in scheduled_activities])
    retrieved = retrieve_similar(query, index, documents)
    stats = get_energy_stats(logs)

    history_text = "\n".join(f"- {r['document']}" for r in retrieved) or "No history yet."
    stats_text = (
        f"- Average energy cost: {stats.get('avg_energy_cost', 'N/A')}/10\n"
        f"- Most draining category: {stats.get('most_draining_category', 'N/A')}\n"
        f"- Days tracked: {stats.get('days_tracked', 0)}"
    ) if stats else "No stats yet."

    activity_lines = [
        f"{i}. {a['activity']} (energy: {a['energy_cost']}/10, category: {a['category']})"
        + (f" — {a['time_slot']}" if a.get("time_slot") else "")
        for i, a in enumerate(scheduled_activities, 1)
    ]

    prompt = SYSTEM_PROMPT + f"""

ACTIVITIES PLANNED FOR TOMORROW:
{chr(10).join(activity_lines)}

PAST ENERGY PATTERNS:
{history_text}

ENERGY STATS:
{stats_text}
"""
    if extra_notes.strip():
        prompt += f"\nADDITIONAL CONTEXT:\n{extra_notes.strip()}"

    prompt += "\n\nPlease generate my plan for tomorrow."

    response = client_gemini.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )
    plan_text = response.text

    # Evaluate output quality
    plan_lower = plan_text.lower()
    high_cost = [a for a in scheduled_activities if a["energy_cost"] >= 7]
    evaluation = {
        "has_recovery_mention": any(w in plan_lower for w in ["recovery", "rest", "buffer", "break", "quiet"]),
        "high_cost_activity_count": len(high_cost),
        "total_estimated_energy": sum(a["energy_cost"] for a in scheduled_activities),
        "plan_length_words": len(plan_text.split()),
        "passed": len(plan_text.split()) <= 400
    }

    # Save plan
    plans = []
    if PLAN_FILE.exists():
        with open(PLAN_FILE, "r") as f:
            plans = json.load(f)
    plans.append({
        "generated_at": datetime.now().isoformat(),
        "scheduled_activities": scheduled_activities,
        "plan_text": plan_text
    })
    with open(PLAN_FILE, "w") as f:
        json.dump(plans, f, indent=2)

    return plan_text, evaluation

# ── UI helpers ─────────────────────────────────────────────────────────────────

planned_activities_state = []

def get_today_log_display():
    logs = load_logs()
    today_logs = [l for l in logs if l["date"] == str(date.today())]
    if not today_logs:
        return "No activities logged today yet."
    lines = []
    total = 0
    for i, l in enumerate(today_logs, 1):
        lines.append(f"**{i}.** {l['activity']} [{l['category']}] — {l['energy_cost']}/10"
                     + (f" | {l['notes']}" if l.get("notes") else ""))
        total += l["energy_cost"]
    lines.append(f"\n**Total energy today: {total} points**")
    return "\n".join(lines)

def log_activity_ui(activity, energy_cost, category, notes):
    if not activity.strip():
        return "⚠️ Please enter an activity name.", get_today_log_display()
    entry = append_activity(activity.strip(), int(energy_cost), category, notes)
    return f"✅ Logged: **{entry['activity']}** — {entry['energy_cost']}/10", get_today_log_display()

def delete_entry_ui(entry_number):
    logs = load_logs()
    today_logs = [l for l in logs if l["date"] == str(date.today())]
    if not today_logs:
        return "⚠️ No entries today.", get_today_log_display()
    try:
        idx = int(entry_number)
        if not 1 <= idx <= len(today_logs):
            return f"⚠️ Invalid number. You have {len(today_logs)} entries today.", get_today_log_display()
        entry_to_delete = today_logs[idx - 1]
        logs = [l for l in logs if l["id"] != entry_to_delete["id"]]
        save_logs(logs)
        return f"🗑️ Deleted: **{entry_to_delete['activity']}**", get_today_log_display()
    except ValueError:
        return "⚠️ Please enter a valid number.", get_today_log_display()

def format_plan_preview():
    if not planned_activities_state:
        return "No activities added yet."
    lines = []
    total = 0
    for i, a in enumerate(planned_activities_state, 1):
        time_str = f" @ {a['time_slot']}" if a.get("time_slot") else ""
        lines.append(f"{i}. {a['activity']}{time_str} [{a['category']}] — {a['energy_cost']}/10")
        total += a["energy_cost"]
    lines.append(f"\n**Estimated total: {total} points**")
    if total > 40:
        lines.append("⚠️ High load — consider removing an activity.")
    return "\n".join(lines)

def add_to_plan_ui(activity, energy_cost, category, time_slot):
    global planned_activities_state
    if not activity.strip():
        return "⚠️ Please enter an activity name.", format_plan_preview()
    planned_activities_state.append({
        "activity": activity.strip(),
        "energy_cost": int(energy_cost),
        "category": category,
        "time_slot": time_slot.strip() if time_slot.strip() else None
    })
    return f"✅ Added: {activity.strip()} ({energy_cost}/10)", format_plan_preview()

def clear_plan_ui():
    global planned_activities_state
    planned_activities_state = []
    return "🗑️ Plan cleared.", format_plan_preview()

def generate_plan_ui(extra_notes):
    global planned_activities_state
    if not planned_activities_state:
        return "⚠️ Add at least one activity to plan first.", ""
    try:
        plan_text, evaluation = generate_plan(planned_activities_state, extra_notes)
        eval_summary = (
            f"📊 **Plan evaluation:**\n"
            f"- Recovery buffer mentioned: {'✅ Yes' if evaluation['has_recovery_mention'] else '⚠️ No'}\n"
            f"- High-cost activities: {evaluation['high_cost_activity_count']}\n"
            f"- Total estimated energy: {evaluation['total_estimated_energy']} points\n"
            f"- Plan length: {evaluation['plan_length_words']} words"
        )
        return plan_text, eval_summary
    except Exception as e:
        return f"❌ Error: {str(e)}", ""

def get_history_display():
    logs = load_logs()
    if not logs:
        return "No history yet."
    stats = get_energy_stats(logs)
    df = pd.DataFrame(logs)
    recent = df.sort_values("timestamp", ascending=False).head(10)
    lines = [
        "## 📊 Your Energy Stats\n",
        f"- Days tracked: {stats.get('days_tracked', 0)}",
        f"- Total activities logged: {stats.get('total_entries', 0)}",
        f"- Average energy cost: {stats.get('avg_energy_cost', 'N/A')}/10",
        f"- Most draining category: {stats.get('most_draining_category', 'N/A')}",
        f"- Most draining activity: {stats.get('most_draining_activity', 'N/A')}",
        "\n## 🕐 10 Most Recent Entries\n"
    ]
    for _, row in recent.iterrows():
        lines.append(f"• [{row['date']}] {row['activity']} ({row['category']}) — {row['energy_cost']}/10")
    return "\n".join(lines)

# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="EnergyPlanner",
    theme=gr.themes.Base(
        primary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"]
    )
) as app:
    gr.Markdown("""# 🔋 EnergyPlanner
**A daily planner that works with your energy, not against it.**
Track what drains you. Plan what protects you.""")

    with gr.Tabs():

        with gr.TabItem("📝 Log Today"):
            with gr.Row():
                with gr.Column(scale=2):
                    log_activity_input = gr.Textbox(label="Activity", placeholder="e.g. Team meeting, grocery run...")
                    log_energy_input = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Energy cost (1=barely anything, 10=completely drained)")
                    log_category_input = gr.Dropdown(choices=ACTIVITY_CATEGORIES, value="Work / Study", label="Category")
                    log_notes_input = gr.Textbox(label="Notes (optional)")
                    log_btn = gr.Button("Log Activity", variant="primary")
                    log_status = gr.Markdown()
                with gr.Column(scale=1):
                    gr.Markdown("### Today's log")
                    today_display = gr.Markdown(value=get_today_log_display)
                    gr.Button("↻ Refresh", variant="secondary").click(fn=get_today_log_display, outputs=today_display)
            log_btn.click(fn=log_activity_ui, inputs=[log_activity_input, log_energy_input, log_category_input, log_notes_input], outputs=[log_status, today_display])

        with gr.TabItem("🗑️ Manage Logs"):
            gr.Markdown("### Delete an entry from today. Check the Log Today tab for entry numbers.")
            with gr.Row():
                with gr.Column(scale=1):
                    delete_number_input = gr.Textbox(label="Entry number to delete", placeholder="e.g. 2")
                    delete_btn = gr.Button("Delete Entry", variant="stop")
                    delete_status = gr.Markdown()
                with gr.Column(scale=1):
                    gr.Markdown("### Today's log")
                    manage_log_display = gr.Markdown(value=get_today_log_display)
                    gr.Button("↻ Refresh", variant="secondary").click(fn=get_today_log_display, outputs=manage_log_display)
            delete_btn.click(fn=delete_entry_ui, inputs=[delete_number_input], outputs=[delete_status, manage_log_display])

        with gr.TabItem("📅 Plan Tomorrow"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("**Step 1 — Add tomorrow's activities**")
                    plan_activity_input = gr.Textbox(label="Activity", placeholder="e.g. Doctor's appointment, dinner with friends...")
                    plan_energy_input = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Expected energy cost")
                    plan_category_input = gr.Dropdown(choices=ACTIVITY_CATEGORIES, value="Work / Study", label="Category")
                    plan_time_input = gr.Textbox(label="Time slot (optional)", placeholder="e.g. 10:00 AM")
                    with gr.Row():
                        add_to_plan_btn = gr.Button("Add to Plan", variant="primary")
                        clear_plan_btn = gr.Button("Clear Plan", variant="secondary")
                    add_status = gr.Markdown()
                    gr.Markdown("**Step 2 — Any context for tomorrow? (optional)**")
                    extra_notes_input = gr.Textbox(label="Notes", placeholder="e.g. rough week, must not cancel morning appointment...", lines=3)
                    gr.Markdown("**Step 3 — Generate**")
                    generate_btn = gr.Button("✨ Generate Tomorrow's Plan", variant="primary")
                with gr.Column(scale=1):
                    gr.Markdown("### Plan preview")
                    plan_preview = gr.Markdown(value=format_plan_preview)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Your plan")
                    plan_output = gr.Markdown()
                with gr.Column():
                    gr.Markdown("### Plan evaluation")
                    eval_output = gr.Markdown()
            add_to_plan_btn.click(fn=add_to_plan_ui, inputs=[plan_activity_input, plan_energy_input, plan_category_input, plan_time_input], outputs=[add_status, plan_preview])
            clear_plan_btn.click(fn=clear_plan_ui, outputs=[add_status, plan_preview])
            generate_btn.click(fn=generate_plan_ui, inputs=[extra_notes_input], outputs=[plan_output, eval_output])

        with gr.TabItem("📊 History & Insights"):
            refresh_history_btn = gr.Button("↻ Load History", variant="secondary")
            history_display = gr.Markdown()
            refresh_history_btn.click(fn=get_history_display, outputs=history_display)

if __name__ == "__main__":
    app.launch()
