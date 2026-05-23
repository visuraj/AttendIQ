"""
AttendIQ — Streamlit App
UI matches the screenshot: Answer on top, then two-panel evidence below
(Attendance Evidence left | Policy Evidence right)
"""

import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from rag_pipeline import AttendIQPipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AttendIQ — Smart Attendance Q&A",
    page_icon="🕐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; }
.stApp                      { background: #f5f6fa; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #1a1d2e;
    color: #e0e0e0;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: #e0e0e0 !important;
}

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #1a1d2e 0%, #2d3561 100%);
    color: white;
    padding: 1.4rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.app-header h1 { margin: 0; font-size: 1.7rem; font-family:'DM Mono',monospace; color:white; }
.app-header p  { margin: 0.2rem 0 0; font-size: 0.85rem; color: #aab4d4; }

/* ── Stat pills ── */
.stats-row { display: flex; gap: 0.8rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
.stat-pill {
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
}
.pill-total    { background:#e8eaf6; color:#3949ab; }
.pill-ontime   { background:#e8f5e9; color:#2e7d32; }
.pill-late     { background:#fff3e0; color:#e65100; }
.pill-early    { background:#fce4ec; color:#c62828; }
.pill-absent   { background:#f3e5f5; color:#6a1b9a; }

/* ── Answer box ── */
.answer-box {
    background: white;
    border: 1px solid #e0e0e0;
    border-left: 4px solid #3949ab;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0 1.5rem;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #1a1d2e;
}
.answer-box .answer-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #9e9e9e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}

/* ── Evidence panels ── */
.panel-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 0.5rem 1rem;
    border-radius: 6px 6px 0 0;
    margin-bottom: 0;
}
.panel-att  { background: #1a1d2e; color: #aab4d4; }
.panel-pol  { background: #2d3561; color: #aab4d4; }

.evidence-card {
    background: white;
    border: 1px solid #e8e8e8;
    border-radius: 0 0 8px 8px;
    padding: 0;
    overflow: hidden;
    margin-bottom: 0.8rem;
}

/* attendance card */
.att-card-header {
    background: #f8f9ff;
    border-bottom: 1px solid #e8eaf6;
    padding: 0.7rem 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.att-doc-id {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    color: #3949ab;
}
.att-badge {
    font-size: 0.68rem;
    font-weight: 600;
    padding: 0.15rem 0.55rem;
    border-radius: 10px;
    font-family: 'DM Mono', monospace;
}
.badge-record   { background:#e8eaf6; color:#3949ab; }
.badge-late     { background:#fff3e0; color:#e65100; }
.badge-early    { background:#fce4ec; color:#c62828; }
.badge-ontime   { background:#e8f5e9; color:#2e7d32; }
.badge-absent   { background:#f3e5f5; color:#6a1b9a; }

.att-body {
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #333;
    line-height: 1.7;
}
.att-body strong { color: #1a1d2e; }

.time-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.4rem;
    margin: 0.5rem 0;
    font-size: 0.8rem;
}
.time-cell {
    background: #f5f6fa;
    border-radius: 4px;
    padding: 0.3rem 0.6rem;
}
.time-label { color: #9e9e9e; font-size: 0.72rem; }
.time-val   { font-family:'DM Mono',monospace; font-weight:500; color:#1a1d2e; }

.flag-row { margin-top: 0.4rem; }
.flag-chip {
    display: inline-block;
    background: #fff3e0;
    color: #e65100;
    font-size: 0.72rem;
    padding: 0.15rem 0.5rem;
    border-radius: 10px;
    font-family: 'DM Mono', monospace;
    margin-right: 0.3rem;
}
.flag-chip.ok { background:#e8f5e9; color:#2e7d32; }
.flag-chip.absent { background:#f3e5f5; color:#6a1b9a; }

/* policy card */
.pol-card {
    background: white;
    border: 1px solid #e8e8e8;
    border-radius: 0 0 8px 8px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.8rem;
}
.pol-doc-id {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #9e9e9e;
    margin-bottom: 0.4rem;
}
.pol-section {
    font-weight: 600;
    font-size: 0.88rem;
    color: #2d3561;
    margin-bottom: 0.4rem;
}
.pol-snippet {
    font-size: 0.82rem;
    color: #555;
    line-height: 1.6;
    border-left: 3px solid #e8eaf6;
    padding-left: 0.7rem;
}

/* search bar */
.stTextInput > div > div > input {
    border-radius: 8px;
    border: 1.5px solid #c5cae9;
    font-size: 0.95rem;
    padding: 0.6rem 1rem;
}

/* button */
.stButton > button {
    background: #3949ab;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.55rem 1.8rem;
}
.stButton > button:hover { background: #283593; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("pipeline", None), ("ready", False), ("history", [])]:
    if k not in st.session_state:
        st.session_state[k] = v

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV    = os.path.join(BASE, "attendance_data.csv")
DEFAULT_POLICY = os.path.join(BASE, "attendance_policy.md")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔑 API Key")
    api_key = st.text_input("Google API Key", type="password", placeholder="sk-...")

    st.markdown("---")
    st.markdown("## 📂 Data")

    csv_file    = st.file_uploader("Upload Attendance CSV", type=["csv"])
    policy_file = st.file_uploader("Upload HR Policy (.md/.txt)", type=["md","txt"])

    if api_key:
        if st.session_state.pipeline is None:
            st.session_state.pipeline = AttendIQPipeline(api_key)
        st.success("✓ API key set")

    if st.button("⚡ Load & Index Data", use_container_width=True):
        if not api_key:
            st.error("Enter API key first.")
        else:
            with st.spinner("Indexing..."):
                pipeline = st.session_state.pipeline

                # Save uploads or use defaults
                csv_path = DEFAULT_CSV
                pol_path = DEFAULT_POLICY

                if csv_file:
                    t = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                    t.write(csv_file.read()); t.flush()
                    csv_path = t.name

                if policy_file:
                    t = tempfile.NamedTemporaryFile(delete=False, suffix=".md")
                    t.write(policy_file.read()); t.flush()
                    pol_path = t.name

                result = pipeline.build_index(csv_path, pol_path)
                st.session_state.ready = True
                st.success(f"✓ {result['attendance_records']} records indexed")

    st.markdown("---")

    # Suggested questions
    st.markdown("## 💡 Try asking")
    questions = [
        "Who was late on 2026-05-18?",
        "Which employees left early?",
        "Who was absent this week?",
        "Show Arjun Nair's attendance",
        "Who came on time every day?",
        "Summarize Engineering team attendance",
    ]
    for q in questions:
        if st.button(q, use_container_width=True, key=q):
            st.session_state["suggested_q"] = q


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <span style="font-size:2rem">🕐</span>
  <div>
    <h1>AttendIQ</h1>
    <p>Smart Employee Attendance Q&amp;A — powered by RAG + GPT-4o-mini</p>
  </div>
</div>
""", unsafe_allow_html=True)

if not api_key:
    st.info("👈 Enter your OpenAI API key in the sidebar, then click **Load & Index Data**.")
    st.stop()

if not st.session_state.ready:
    st.info("👈 Click **Load & Index Data** in the sidebar to get started.")
    st.stop()

# Stats bar
summary = st.session_state.pipeline.get_summary()
st.markdown(f"""
<div class="stats-row">
  <span class="stat-pill pill-total">📋 {summary.get('total',0)} Records</span>
  <span class="stat-pill pill-ontime">✅ {summary.get('on_time',0)} On Time</span>
  <span class="stat-pill pill-late">⏰ {summary.get('late',0)} Late</span>
  <span class="stat-pill pill-early">🚪 {summary.get('early_departure',0)} Early Out</span>
  <span class="stat-pill pill-absent">❌ {summary.get('absent',0)} Absent</span>
</div>
""", unsafe_allow_html=True)

# Query input
default_q = st.session_state.pop("suggested_q", "")
col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input(
        "Ask", value=default_q,
        placeholder="e.g. Who was late on 2026-05-18?",
        label_visibility="collapsed"
    )
with col2:
    ask = st.button("Ask →", use_container_width=True)

if ask and question.strip():
    with st.spinner("Retrieving evidence & generating answer..."):
        result = st.session_state.pipeline.query(question)
        st.session_state.history.insert(0, {"q": question, **result})

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.history:
    latest = st.session_state.history[0]

    # Answer box
    st.markdown(f"""
    <div class="answer-box">
      <div class="answer-label">Answer</div>
      {latest['answer']}
    </div>
    """, unsafe_allow_html=True)

    # Two-panel evidence
    col_att, col_pol = st.columns(2)

    # ── Attendance Evidence ────────────────────────────────────────────────────
    with col_att:
        st.markdown('<div class="panel-title panel-att">📋 Attendance Evidence</div>', unsafe_allow_html=True)

        if not latest["attendance"]:
            st.markdown('<div class="evidence-card" style="padding:1rem;color:#999">No records found.</div>', unsafe_allow_html=True)
        else:
            for ev in latest["attendance"]:
                verdict  = ev.get("verdict", "")
                flags    = ev.get("flags", "")
                status   = ev.get("status", "present")

                badge_cls = "badge-ontime"
                if "late" in verdict:   badge_cls = "badge-late"
                if "early" in verdict:  badge_cls = "badge-early"
                if status == "absent":  badge_cls = "badge-absent"

                flag_chips = ""
                for f in flags.split(","):
                    f = f.strip()
                    if not f: continue
                    css = "ok" if f == "on time" else ("absent" if f == "absent" else "")
                    flag_chips += f'<span class="flag-chip {css}">{f}</span>'

                ai = ev.get("actual_in", "—") or "—"
                ao = ev.get("actual_out","—") or "—"

                st.markdown(f"""
                <div style="margin-bottom:0.8rem">
                  <div class="panel-title panel-att" style="border-radius:6px 6px 0 0; display:flex; justify-content:space-between; align-items:center;">
                    <span class="att-doc-id">{ev['doc_id']}</span>
                    <span class="att-badge badge-record">record</span>
                  </div>
                  <div class="att-body" style="border:1px solid #e8e8e8; border-top:none; border-radius:0 0 8px 8px; background:white;">
                    <strong>{ev['name']}</strong> ({ev['department']}) on {ev['date']}:<br>
                    <div class="time-grid">
                      <div class="time-cell"><div class="time-label">Scheduled In</div><div class="time-val">{ev['scheduled_in']}</div></div>
                      <div class="time-cell"><div class="time-label">Scheduled Out</div><div class="time-val">{ev['scheduled_out']}</div></div>
                      <div class="time-cell"><div class="time-label">Actual In</div><div class="time-val">{ai}</div></div>
                      <div class="time-cell"><div class="time-label">Actual Out</div><div class="time-val">{ao}</div></div>
                    </div>
                    status: <span class="att-badge {badge_cls}">{status}</span><br>
                    <div class="flag-row">Flags: {flag_chips}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Policy Evidence ────────────────────────────────────────────────────────
    with col_pol:
        st.markdown('<div class="panel-title panel-pol">📜 Policy Evidence</div>', unsafe_allow_html=True)

        if not latest["policy"]:
            st.markdown('<div class="pol-card" style="color:#999">No relevant policy found.</div>', unsafe_allow_html=True)
        else:
            for pv in latest["policy"]:
                st.markdown(f"""
                <div class="pol-card">
                  <div class="pol-doc-id">{pv['doc_id']}</div>
                  <div class="pol-section">## {pv['section']}</div>
                  <div class="pol-snippet">{pv['snippet']}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── History ────────────────────────────────────────────────────────────────
    if len(st.session_state.history) > 1:
        st.markdown("---")
        st.markdown("### 🕑 Previous Queries")
        for item in st.session_state.history[1:]:
            with st.expander(f"Q: {item['q'][:80]}"):
                st.markdown(f'<div class="answer-box"><div class="answer-label">Answer</div>{item["answer"]}</div>', unsafe_allow_html=True)
                for ev in item.get("attendance", []):
                    st.markdown(f"- `{ev['doc_id']}` — {ev['name']}, {ev['date']}, flags: {ev['flags']}")
