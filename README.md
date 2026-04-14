# ✈️ American Airlines HR Learning & Development Intelligence System

An AI-powered HR L&D recommendation platform built with **Streamlit**, **LLaMA (via HuggingFace)**, and **synthetic American Airlines data**.

---

## 🏗️ Project Structure

```
aa_hr_ld/
├── app.py                    # Main Streamlit application
├── requirements.txt
├── agent/
│   ├── __init__.py
│   └── hr_agent.py           # LlamaClient + HRAgent logic
└── data/
    ├── __init__.py
    └── synthetic_data.py     # 200 synthetic AA employees + courses + gaps
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get HuggingFace API Token
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new **Read** token
3. Visit [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) and **request access** (free, approved instantly)

### 3. Run the app
```bash
streamlit run app.py
```

### 4. In the sidebar:
- Paste your HuggingFace token
- Select a LLaMA model (3B = fastest, 8B = better quality)
- Click **Initialize Agent**
- Pick or type an Employee ID (e.g. `AA00042`)
- Click **Load Employee Profile**

---

## 🤖 AI Features

| Feature | Description |
|---|---|
| **Alex AI Advisor** | Conversational L&D advisor powered by LLaMA |
| **L&D Recommendations** | Personalized course recommendations per employee |
| **Skill Gap Analysis** | Prioritized gap breakdown with learning actions |
| **Career Path Options** | 2-3 tailored career trajectories at American Airlines |
| **30-60-90 Day Plans** | Structured development roadmaps |
| **Free-form Chat** | Ask anything about learning, careers, certifications |

---

## 📊 Analytics Tabs

- **Analytics Dashboard** — KPIs, dept learning hours, performance distribution, skill gap heatmap
- **Course Catalog** — 25 AA courses filterable by category, format, level
- **Employee Explorer** — Browse/search 200 synthetic employees
- **Skill Gap Map** — Heatmap of skills × departments

---

## 🔧 LLaMA Models Available

| Model Key | Model ID | Notes |
|---|---|---|
| `llama-3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` | Fast, free-tier friendly ✅ |
| `llama-3.1-8b` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Better quality |
| `llama-3-8b` | `meta-llama/Meta-Llama-3-8B-Instruct` | Stable alternative |

---

## 🧪 Synthetic Data Details

| Dataset | Size | Description |
|---|---|---|
| Employees | 200 | 12 depts, all AA roles, performance, skills, goals |
| Courses | 25 | Real AA + industry courses across 8 categories |
| Learning History | ~600 records | Completions with scores & feedback |
| Skill Gaps | ~800 records | Prioritized gaps per employee |

---

## 💡 Example Questions for Alex

- *"What are the top 3 courses I should take this quarter?"*
- *"I want to move into management — what should I do?"*
- *"Which certifications are most valuable for a Safety Officer?"*
- *"Create a learning plan that fits my 4 hours/week schedule"*
- *"How does my learning compare to others in my department?"*

---

## 🛠️ Customization

**Add real data**: Replace `get_dataframes()` calls in `hr_agent.py` with your HRIS/LMS data source.

**Switch LLM provider**: Modify `LlamaClient` to use OpenAI, Groq, Ollama, or any OpenAI-compatible API.

**Add tools**: Extend `HRAgent` with calendar booking, LMS enrollment, or Workday integration.