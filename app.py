"""
American Airlines HR Learning & Development Recommendation System
Streamlit Application
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from agent.hr_agent import HRAgent, LlamaClient
from data.synthetic_data import get_dataframes, COURSES, DEPARTMENTS

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AA HR L&D Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --aa-red: #E31837;
    --aa-navy: #0D1B2A;
    --aa-blue: #1B4F9B;
    --aa-light: #F7F9FC;
    --aa-silver: #C8D1DC;
    --aa-gold: #C9A84C;
    --aa-green: #1A7F4B;
    --radius: 12px;
}

/* Global */
.stApp { background: var(--aa-navy); }
section[data-testid="stSidebar"] { background: #080F1A !important; border-right: 1px solid #1e2d40; }
.block-container { padding-top: 1.5rem; }

/* Fonts */
h1, h2, h3 { font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 1px; }
p, span, div, label, .stMarkdown { font-family: 'Inter', sans-serif !important; }

/* Header banner */
.aa-header {
    background: linear-gradient(135deg, #E31837 0%, #8B0F23 50%, #0D1B2A 100%);
    border-radius: var(--radius);
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    border-bottom: 3px solid var(--aa-gold);
    display: flex; align-items: center; gap: 1rem;
}
.aa-header h1 { color: white; font-size: 2.2rem; margin: 0; }
.aa-header p { color: rgba(255,255,255,0.75); margin: 0; font-size: 0.9rem; }

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #0f1e30 0%, #152840 100%);
    border: 1px solid #1e3450;
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-2px); border-color: var(--aa-red); }
.metric-card .value { font-size: 2rem; font-weight: 700; color: var(--aa-gold); font-family: 'Bebas Neue', sans-serif !important; }
.metric-card .label { font-size: 0.78rem; color: var(--aa-silver); text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

/* Employee card */
.emp-card {
    background: linear-gradient(135deg, #0f1e30, #152840);
    border: 1px solid var(--aa-blue);
    border-left: 4px solid var(--aa-red);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.emp-name { font-size: 1.4rem; color: white; font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 1px; }
.emp-role { color: var(--aa-gold); font-size: 0.85rem; margin-bottom: 0.5rem; }
.emp-dept { color: var(--aa-silver); font-size: 0.8rem; }

/* Chat */
.chat-container {
    background: #080F1A;
    border: 1px solid #1e3450;
    border-radius: var(--radius);
    padding: 1rem;
    max-height: 480px;
    overflow-y: auto;
}
.msg-user {
    background: linear-gradient(135deg, #1B4F9B, #0d2f6e);
    border-radius: 12px 12px 2px 12px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0 0.5rem 3rem;
    color: white; font-size: 0.88rem;
}
.msg-alex {
    background: linear-gradient(135deg, #0f1e30, #152840);
    border: 1px solid #1e3450;
    border-left: 3px solid var(--aa-red);
    border-radius: 2px 12px 12px 12px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 3rem 0.5rem 0;
    color: #e2e8f0; font-size: 0.88rem;
}
.msg-meta { font-size: 0.7rem; color: var(--aa-silver); margin-bottom: 4px; }

/* Pills / Tags */
.skill-pill {
    display: inline-block;
    background: rgba(27, 79, 155, 0.25);
    border: 1px solid rgba(27, 79, 155, 0.5);
    color: #90c4ff;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    margin: 2px;
}
.gap-pill-high {
    display: inline-block;
    background: rgba(227, 24, 55, 0.2);
    border: 1px solid rgba(227, 24, 55, 0.5);
    color: #ff8fa3;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    margin: 2px;
}
.gap-pill-med {
    display: inline-block;
    background: rgba(201, 168, 76, 0.2);
    border: 1px solid rgba(201, 168, 76, 0.5);
    color: #f0d080;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    margin: 2px;
}

/* Course card */
.course-card {
    background: #0f1e30;
    border: 1px solid #1e3450;
    border-radius: var(--radius);
    padding: 1rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}
.course-card:hover { border-color: var(--aa-gold); }
.course-title { color: white; font-weight: 600; font-size: 0.9rem; }
.course-meta { color: var(--aa-silver); font-size: 0.75rem; margin-top: 4px; }
.course-free { color: var(--aa-green); font-weight: 600; font-size: 0.75rem; }
.course-cost { color: var(--aa-gold); font-weight: 600; font-size: 0.75rem; }

/* Quick action buttons */
.stButton > button {
    background: linear-gradient(135deg, #1B4F9B, #0d2f6e) !important;
    color: white !important;
    border: 1px solid rgba(27, 79, 155, 0.5) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #E31837, #8B0F23) !important;
    border-color: var(--aa-red) !important;
    transform: translateY(-1px) !important;
}

/* Section headers */
.section-header {
    border-bottom: 2px solid var(--aa-red);
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
    color: white;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    letter-spacing: 1px;
}

/* Alerts */
.info-box {
    background: rgba(27, 79, 155, 0.15);
    border: 1px solid rgba(27, 79, 155, 0.4);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #90c4ff;
    font-size: 0.85rem;
    margin-bottom: 0.75rem;
}

/* Sidebar labels */
.sidebar-label {
    color: var(--aa-silver);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 4px;
}

/* Plotly chart background fix */
.js-plotly-plot { border-radius: var(--radius); }

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* Input styling */
.stTextInput > div > div > input, .stSelectbox > div > div > select {
    background: #0f1e30 !important;
    border: 1px solid #1e3450 !important;
    color: white !important;
    border-radius: 8px !important;
}
.stTextArea textarea {
    background: #0f1e30 !important;
    border: 1px solid #1e3450 !important;
    color: white !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "employee_loaded" not in st.session_state:
    st.session_state.employee_loaded = False
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Chat"
if "data" not in st.session_state:
    st.session_state.data = get_dataframes(200)

data = st.session_state.data

# ─── HEADER ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="aa-header">
    <div>
        <h1>✈️ AA LEARNING & DEVELOPMENT HUB</h1>
        <p>AI-Powered Career Development · American Airlines · Powered by LLaMA on HuggingFace</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    hf_token = st.text_input(
        "HuggingFace API Token",
        type="password",
        placeholder="hf_xxxxxxxxxxxx",
        help="Get your free token at huggingface.co/settings/tokens. Ensure you've requested access to meta-llama models.",
    )

    model_choice = st.selectbox(
        "LLaMA Model",
        options=["llama-3.2-3b", "llama-3.1-8b", "llama-3-8b"],
        help="3B is fastest & free-tier friendly. 8B gives better results.",
    )

    if hf_token and st.button("🔌 Initialize Agent", use_container_width=True):
        with st.spinner("Initializing LLaMA agent..."):
            llm = LlamaClient(hf_token=hf_token, model_key=model_choice)
            st.session_state.agent = HRAgent(llm=llm)
            st.success("Agent ready!")

    st.markdown("---")
    st.markdown("### 👤 Employee Lookup")

    # Quick employee picker
    emp_df = data["employees"]
    sample_ids = emp_df["employee_id"].sample(10).tolist()
    emp_id_select = st.selectbox("Quick-select Employee", ["— type or pick —"] + sample_ids)

    emp_id_input = st.text_input("Or enter Employee ID", placeholder="AA00042")

    final_emp_id = emp_id_input if emp_id_input else (emp_id_select if emp_id_select != "— type or pick —" else "")

    if final_emp_id and st.session_state.agent:
        if st.button("📂 Load Employee Profile", use_container_width=True):
            with st.spinner("Loading profile..."):
                success, intro = st.session_state.agent.load_employee(final_emp_id)
                if success:
                    st.session_state.employee_loaded = True
                    st.session_state.messages = [{"role": "assistant", "content": intro}]
                    st.success(f"Loaded {final_emp_id}")
                else:
                    st.error(intro)
    elif final_emp_id and not st.session_state.agent:
        st.warning("Initialize the agent first.")

    st.markdown("---")
    st.markdown("### 🗃️ Data Overview")
    st.metric("Total Employees", len(emp_df))
    st.metric("Courses Available", len(COURSES))
    gaps_df = data["skill_gaps"]
    st.metric("Skill Gaps Identified", len(gaps_df))

    st.markdown("---")
    st.caption("© 2025 American Airlines HR Analytics")
    st.caption("Powered by LLaMA · HuggingFace · Streamlit")

# ─── MAIN TABS ────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 AI Advisor (Alex)", "📊 Analytics Dashboard", 
    "📚 Course Catalog", "👥 Employee Explorer",
    "🗺️ Skill Gap Map"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — AI Advisor
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    if not st.session_state.agent:
        st.markdown("""
        <div class="info-box">
            🔑 <strong>Getting Started:</strong> Enter your HuggingFace API token in the sidebar, 
            click <em>Initialize Agent</em>, then load an employee profile to begin.
            <br><br>
            <strong>HuggingFace Setup:</strong><br>
            1. Go to <code>huggingface.co/settings/tokens</code> → Create a new token (read access)<br>
            2. Visit <code>huggingface.co/meta-llama/Llama-3.2-3B-Instruct</code> → Request access<br>
            3. Paste your token above and select a model
        </div>
        """, unsafe_allow_html=True)

    # Employee profile card
    if st.session_state.employee_loaded and st.session_state.agent:
        emp = st.session_state.agent.current_employee
        skills_html = "".join([f'<span class="skill-pill">{s}</span>' for s in emp.get("current_skills", [])])
        
        gaps = st.session_state.agent.get_employee_skill_gaps(emp["employee_id"])
        high_gaps = [g for g in gaps if g.get("priority") == "High"]
        gap_html = "".join([f'<span class="gap-pill-high">{g["skill"]}</span>' for g in high_gaps[:5]])
        
        perf_color = {"Outstanding": "#C9A84C", "Exceeds Expectations": "#1A7F4B", 
                      "Meets Expectations": "#1B4F9B", "Needs Improvement": "#E31837"}.get(emp.get("performance_rating", ""), "#90c4ff")

        st.markdown(f"""
        <div class="emp-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <div class="emp-name">{emp['employee_id']} · {emp['job_title']}</div>
                    <div class="emp-role">{emp['department']}</div>
                    <div class="emp-dept">📍 {emp['location']} &nbsp;|&nbsp; 🗓️ {emp['years_at_aa']} yrs &nbsp;|&nbsp; 📚 {emp['learning_hours_ytd']}h YTD</div>
                </div>
                <div style="text-align:right;">
                    <span style="background:rgba(0,0,0,0.3); border:1px solid {perf_color}; color:{perf_color}; padding:4px 12px; border-radius:20px; font-size:0.75rem;">
                        {emp['performance_rating']}
                    </span>
                    <div style="color:#90c4ff; font-size:0.78rem; margin-top:6px;">🎯 {emp['career_goal']}</div>
                </div>
            </div>
            <div style="margin-top:0.75rem;">
                <div style="font-size:0.7rem; color:#8096b0; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">Current Skills</div>
                {skills_html}
            </div>
            {f'<div style="margin-top:0.5rem;"><div style="font-size:0.7rem; color:#8096b0; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">High-Priority Gaps</div>{gap_html}</div>' if gap_html else ''}
        </div>
        """, unsafe_allow_html=True)

    # Quick Analysis Buttons
    if st.session_state.employee_loaded:
        st.markdown('<div class="section-header">⚡ Quick Analysis</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        quick_actions = {
            col1: ("📋 L&D Recommendations", "recommendations"),
            col2: ("🔍 Skill Gap Analysis", "skill_gaps"),
            col3: ("🚀 Career Path Options", "career_path"),
            col4: ("📅 30-60-90 Day Plan", "30_60_90"),
        }
        
        for col, (label, action_key) in quick_actions.items():
            with col:
                if st.button(label, use_container_width=True):
                    with st.spinner(f"Alex is analyzing..."):
                        result = st.session_state.agent.get_quick_analysis(action_key)
                        st.session_state.messages.append({"role": "assistant", "content": result})
                        st.rerun()

    # ── Chat Interface ──
    st.markdown('<div class="section-header">💬 Chat with Alex</div>', unsafe_allow_html=True)
    
    chat_html = '<div class="chat-container">'
    if not st.session_state.messages:
        chat_html += '<div style="color:#4a6080; text-align:center; padding:2rem;">Load an employee profile and start chatting with Alex 👋</div>'
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += f'<div class="msg-user"><div class="msg-meta">You</div>{msg["content"]}</div>'
            else:
                content = msg["content"].replace("\n", "<br>")
                chat_html += f'<div class="msg-alex"><div class="msg-meta">🤖 Alex · AA L&D Advisor</div>{content}</div>'
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

    # Input area
    col_input, col_send = st.columns([5, 1])
    with col_input:
        user_input = st.text_area(
            "Message Alex",
            placeholder="Ask about course recommendations, career paths, skill gaps, certifications...",
            height=80,
            label_visibility="collapsed",
            key="chat_input",
        )
    with col_send:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        send = st.button("Send ✈️", use_container_width=True)

    # Suggested prompts
    st.markdown("**💡 Try asking:**")
    sugg_cols = st.columns(4)
    suggestions = [
        "What courses should I take next?",
        "How can I become a manager?",
        "What certifications are valuable in my field?",
        "Create a learning plan for Q3",
    ]
    for i, (col, sug) in enumerate(zip(sugg_cols, suggestions)):
        with col:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                if st.session_state.agent and st.session_state.employee_loaded:
                    with st.spinner("Alex is thinking..."):
                        response = st.session_state.agent.chat(sug)
                        st.session_state.messages.append({"role": "user", "content": sug})
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.rerun()

    if send and user_input.strip():
        if not st.session_state.agent:
            st.error("Please initialize the agent first.")
        elif not st.session_state.employee_loaded:
            st.warning("Please load an employee profile first.")
        else:
            with st.spinner("Alex is thinking..."):
                response = st.session_state.agent.chat(user_input.strip())
                st.session_state.messages.append({"role": "user", "content": user_input.strip()})
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

    if st.session_state.messages and st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analytics Dashboard
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-header">📊 L&D Analytics Dashboard</div>', unsafe_allow_html=True)

    employees_df = data["employees"]
    history_df = data["learning_history"]
    gaps_df = data["skill_gaps"]

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    kpis = [
        (len(employees_df), "Total Employees"),
        (f"{employees_df['learning_hours_ytd'].mean():.1f}h", "Avg Learning Hours YTD"),
        (len(gaps_df), "Total Skill Gaps"),
        (f"{len(gaps_df[gaps_df['priority']=='High'])}", "High Priority Gaps"),
        (f"{(len(history_df) / len(employees_df)):.1f}", "Avg Courses/Employee"),
    ]
    for col, (val, label) in zip([col1, col2, col3, col4, col5], kpis):
        with col:
            st.markdown(f'<div class="metric-card"><div class="value">{val}</div><div class="label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Dept Learning Hours + Performance Ratings
    col_left, col_right = st.columns(2)

    with col_left:
        dept_hours = employees_df.groupby("department")["learning_hours_ytd"].mean().sort_values(ascending=True).reset_index()
        fig = px.bar(
            dept_hours, x="learning_hours_ytd", y="department", orientation="h",
            title="Avg Learning Hours YTD by Department",
            color="learning_hours_ytd", color_continuous_scale=["#1B4F9B", "#E31837"],
            labels={"learning_hours_ytd": "Hours", "department": ""},
        )
        fig.update_layout(
            plot_bgcolor="#0f1e30", paper_bgcolor="#0f1e30",
            font_color="white", showlegend=False,
            coloraxis_showscale=False,
            title_font=dict(size=14, color="#C9A84C"),
            height=320,
        )
        fig.update_xaxes(gridcolor="#1e3450", color="#8096b0")
        fig.update_yaxes(color="#c8d1dc")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        perf_counts = employees_df["performance_rating"].value_counts().reset_index()
        perf_counts.columns = ["rating", "count"]
        colors = {"Outstanding": "#C9A84C", "Exceeds Expectations": "#1A7F4B",
                  "Meets Expectations": "#1B4F9B", "Needs Improvement": "#E31837"}
        fig2 = px.pie(
            perf_counts, values="count", names="rating",
            title="Performance Rating Distribution",
            color="rating",
            color_discrete_map=colors,
            hole=0.45,
        )
        fig2.update_layout(
            plot_bgcolor="#0f1e30", paper_bgcolor="#0f1e30",
            font_color="white", title_font=dict(size=14, color="#C9A84C"),
            legend=dict(font=dict(color="white", size=11)),
            height=320,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Skill Gaps by Department + Course Popularity
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        dept_gaps = gaps_df.merge(employees_df[["employee_id", "department"]], on="employee_id")
        gap_summary = dept_gaps.groupby(["department", "priority"]).size().reset_index(name="count")
        fig3 = px.bar(
            gap_summary, x="department", y="count", color="priority",
            title="Skill Gaps by Department & Priority",
            color_discrete_map={"High": "#E31837", "Medium": "#C9A84C", "Low": "#1B4F9B"},
            barmode="stack",
        )
        fig3.update_layout(
            plot_bgcolor="#0f1e30", paper_bgcolor="#0f1e30",
            font_color="white", title_font=dict(size=14, color="#C9A84C"),
            legend=dict(font=dict(color="white", size=11)),
            height=340, xaxis_tickangle=-35,
        )
        fig3.update_xaxes(gridcolor="#1e3450", color="#8096b0")
        fig3.update_yaxes(gridcolor="#1e3450", color="#8096b0")
        st.plotly_chart(fig3, use_container_width=True)

    with col_right2:
        course_pop = history_df.groupby("course_title")["employee_id"].count().sort_values(ascending=False).head(8).reset_index()
        course_pop.columns = ["course", "enrollments"]
        course_pop["course"] = course_pop["course"].str[:35] + "..."
        fig4 = px.bar(
            course_pop, x="enrollments", y="course", orientation="h",
            title="Top 8 Most Popular Courses",
            color="enrollments", color_continuous_scale=["#1B4F9B", "#C9A84C"],
        )
        fig4.update_layout(
            plot_bgcolor="#0f1e30", paper_bgcolor="#0f1e30",
            font_color="white", showlegend=False,
            coloraxis_showscale=False,
            title_font=dict(size=14, color="#C9A84C"), height=340,
        )
        fig4.update_xaxes(gridcolor="#1e3450", color="#8096b0")
        fig4.update_yaxes(color="#c8d1dc")
        st.plotly_chart(fig4, use_container_width=True)

    # Row 3: Learning Hours vs Performance
    perf_learn = employees_df.groupby("performance_rating")["learning_hours_ytd"].mean().reset_index()
    fig5 = px.bar(
        perf_learn, x="performance_rating", y="learning_hours_ytd",
        title="Avg Learning Hours by Performance Rating — Correlation Analysis",
        color="performance_rating",
        color_discrete_map={"Outstanding": "#C9A84C", "Exceeds Expectations": "#1A7F4B",
                            "Meets Expectations": "#1B4F9B", "Needs Improvement": "#E31837"},
    )
    fig5.update_layout(
        plot_bgcolor="#0f1e30", paper_bgcolor="#0f1e30",
        font_color="white", showlegend=False,
        title_font=dict(size=14, color="#C9A84C"), height=280,
    )
    fig5.update_xaxes(gridcolor="#1e3450", color="#8096b0")
    fig5.update_yaxes(gridcolor="#1e3450", color="#8096b0", title="Avg Hours")
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Course Catalog
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-header">📚 Course Catalog</div>', unsafe_allow_html=True)

    courses_df = pd.DataFrame(COURSES)

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        cat_filter = st.multiselect("Category", courses_df["category"].unique(), placeholder="All categories")
    with fc2:
        format_filter = st.multiselect("Format", courses_df["format"].unique(), placeholder="All formats")
    with fc3:
        level_filter = st.multiselect("Level", courses_df["level"].unique(), placeholder="All levels")

    filtered = courses_df.copy()
    if cat_filter:
        filtered = filtered[filtered["category"].isin(cat_filter)]
    if format_filter:
        filtered = filtered[filtered["format"].isin(format_filter)]
    if level_filter:
        filtered = filtered[filtered["level"].isin(level_filter)]

    st.markdown(f"**{len(filtered)} courses** matching your filters")

    level_colors = {"Foundational": "#1A7F4B", "Intermediate": "#1B4F9B", "Advanced": "#E31837"}

    for _, course in filtered.iterrows():
        lc = level_colors.get(course["level"], "#8096b0")
        cost_html = f'<span class="course-free">FREE</span>' if course["cost"] == 0 else f'<span class="course-cost">${course["cost"]}</span>'
        st.markdown(f"""
        <div class="course-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <div class="course-title">{course["id"]} · {course["title"]}</div>
                    <div class="course-meta">
                        🏷️ {course["category"]} &nbsp;|&nbsp; 
                        ⏱️ {course["duration_hours"]}h &nbsp;|&nbsp;
                        📺 {course["format"]} &nbsp;|&nbsp;
                        🏫 {course["provider"]}
                    </div>
                </div>
                <div style="text-align:right; flex-shrink:0; margin-left:1rem;">
                    <span style="background:rgba(0,0,0,0.3); border:1px solid {lc}; color:{lc}; padding:3px 10px; border-radius:12px; font-size:0.72rem;">{course["level"]}</span>
                    <div style="margin-top:6px;">{cost_html}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Employee Explorer
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-header">👥 Employee Explorer</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        dept_f = st.selectbox("Department", ["All"] + sorted(DEPARTMENTS))
    with col_f2:
        perf_f = st.selectbox("Performance", ["All", "Outstanding", "Exceeds Expectations", "Meets Expectations", "Needs Improvement"])
    with col_f3:
        search_id = st.text_input("Search by ID", placeholder="AA00042")

    disp = data["employees"].copy()
    if dept_f != "All":
        disp = disp[disp["department"] == dept_f]
    if perf_f != "All":
        disp = disp[disp["performance_rating"] == perf_f]
    if search_id:
        disp = disp[disp["employee_id"].str.contains(search_id.upper())]

    st.markdown(f"**{len(disp)} employees** found")

    display_cols = ["employee_id", "department", "job_title", "job_level", "location",
                    "years_at_aa", "performance_rating", "learning_hours_ytd", "career_goal"]
    
    st.dataframe(
        disp[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=420,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Skill Gap Map
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown('<div class="section-header">🗺️ Skill Gap Heat Map</div>', unsafe_allow_html=True)

    gaps_df = data["skill_gaps"]
    employees_df = data["employees"]

    dept_skill_gaps = (
        gaps_df
        .merge(employees_df[["employee_id", "department"]], on="employee_id")
        .groupby(["department", "skill"])
        .size()
        .reset_index(name="gap_count")
    )

    # Top skills with most gaps
    top_skills = dept_skill_gaps.groupby("skill")["gap_count"].sum().nlargest(15).index.tolist()
    pivot_data = dept_skill_gaps[dept_skill_gaps["skill"].isin(top_skills)]
    pivot = pivot_data.pivot_table(index="department", columns="skill", values="gap_count", fill_value=0)

    fig_heat = px.imshow(
        pivot,
        title="Skill Gap Heatmap: Departments × Top 15 Skills",
        color_continuous_scale=["#0f1e30", "#1B4F9B", "#E31837"],
        aspect="auto",
    )
    fig_heat.update_layout(
        plot_bgcolor="#0f1e30", paper_bgcolor="#0f1e30",
        font_color="white", title_font=dict(size=14, color="#C9A84C"),
        height=420,
        xaxis=dict(tickangle=-40, color="#c8d1dc"),
        yaxis=dict(color="#c8d1dc"),
        coloraxis_colorbar=dict(tickcolor="white", titlefont=dict(color="white")),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Top gaps table
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Top 10 Organization-Wide Skill Gaps**")
        top_org_gaps = gaps_df["skill"].value_counts().head(10).reset_index()
        top_org_gaps.columns = ["Skill", "# Employees with Gap"]
        st.dataframe(top_org_gaps, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("**High Priority Gaps by Department**")
        high_p = (
            gaps_df[gaps_df["priority"] == "High"]
            .merge(employees_df[["employee_id", "department"]], on="employee_id")
            .groupby("department")
            .size()
            .sort_values(ascending=False)
            .reset_index()
        )
        high_p.columns = ["Department", "High Priority Gaps"]
        st.dataframe(high_p, use_container_width=True, hide_index=True)