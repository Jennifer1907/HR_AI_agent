"""
HR Learning & Development Agent
Uses LLaMA via HuggingFace Inference API
"""

import os
import json
import re
from typing import Optional
import requests
import pandas as pd

from data.synthetic_data import get_dataframes, COURSES, SKILLS


# ─── HuggingFace LLaMA Client ─────────────────────────────────────────────────

class LlamaClient:
    MODELS = {
        "llama-3.2-3b":  "meta-llama/Llama-3.2-3B-Instruct",
        "llama-3.1-8b":  "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
        "qwen-2.5-7b":   "Qwen/Qwen2.5-7B-Instruct",
        "mistral-7b":    "mistralai/Mistral-7B-Instruct-v0.3",
    }

    def __init__(self, hf_token: str, model_key: str = "llama-3.2-3b"):
        self.token = hf_token
        self.model_key = model_key
        self.model_id = self.MODELS.get(model_key, self.MODELS["llama-3.2-3b"])
        self.api_url = "https://router.huggingface.co/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json",
        }

    def chat(self, messages, max_tokens=1024, temperature=0.7, system_prompt=None):
        payload_messages = []
        if system_prompt:
            payload_messages.append({"role": "system", "content": system_prompt})
        payload_messages.extend(messages)

        payload = {
            "model": self.model_id,
            "messages": payload_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=60
        )

        print("status:", response.status_code)
        print("body:", response.text)

        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]


# ─── HR Agent ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Alex, an expert HR Learning & Development Advisor for American Airlines. 
You help employees grow their careers through personalized learning recommendations, skill gap analysis, 
and development planning. You have deep knowledge of aviation industry training requirements, 
FAA regulations, and American Airlines' internal programs.

Your personality:
- Warm, encouraging, and professional
- Data-driven but human-centered
- Knowledgeable about aviation career paths
- Concise and actionable in your responses

When making recommendations:
1. Always reference the employee's current skills, goals, and performance
2. Prioritize mandatory safety/compliance training when relevant
3. Suggest both immediate wins and long-term development
4. Consider learning format preferences
5. Balance technical skills with soft skills

American Airlines core values: Caring, Integrity, Passion, Safety.
Always align development recommendations with these values."""


class HRAgent:
    def __init__(self, llm: LlamaClient):
        self.llm = llm
        self.data = get_dataframes(200)
        self.conversation_history: list[dict] = []
        self.current_employee: Optional[dict] = None

    # ─── Data Access ──────────────────────────────────────────────────────────

    def get_employee(self, employee_id: str) -> Optional[dict]:
        emp_df = self.data["employees"]
        match = emp_df[emp_df["employee_id"] == employee_id.upper()]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    def get_employee_skill_gaps(self, employee_id: str) -> list[dict]:
        gaps_df = self.data["skill_gaps"]
        gaps = gaps_df[gaps_df["employee_id"] == employee_id.upper()]
        return gaps.to_dict("records")

    def get_employee_learning_history(self, employee_id: str) -> list[dict]:
        hist_df = self.data["learning_history"]
        history = hist_df[hist_df["employee_id"] == employee_id.upper()]
        return history.to_dict("records")

    def get_recommended_courses(self, employee: dict, max_courses: int = 5) -> list[dict]:
        """Rule-based pre-filtering before LLM enrichment."""
        completed = set(employee.get("completed_courses", []))
        dept = employee.get("department", "")
        goal = employee.get("career_goal", "")
        pref_format = employee.get("preferred_format", "Online")

        scored = []
        for course in COURSES:
            if course["id"] in completed:
                continue

            score = 0

            # Category match
            cat_lower = course["category"].lower()
            dept_lower = dept.lower()
            if any(kw in dept_lower for kw in cat_lower.split()):
                score += 3
            if "safety" in dept_lower and "safety" in cat_lower:
                score += 5  # Safety always prioritized
            if "management" in goal.lower() and "leadership" in cat_lower:
                score += 4
            if "data" in goal.lower() and "technology" in cat_lower:
                score += 4
            if "certification" in goal.lower() and "prep" in course["title"].lower():
                score += 4

            # Format preference
            if course["format"] == pref_format or course["format"] == "Online":
                score += 1

            # Cost preference (free = more accessible)
            if course["cost"] == 0:
                score += 1

            scored.append((score, course))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:max_courses]]

    # ─── Context Builder ──────────────────────────────────────────────────────

    def _build_employee_context(self, employee: dict) -> str:
        gaps = self.get_employee_skill_gaps(employee["employee_id"])
        history = self.get_employee_learning_history(employee["employee_id"])
        recommended = self.get_recommended_courses(employee)

        high_priority_gaps = [g["skill"] for g in gaps if g.get("priority") == "High"]
        recent_courses = [h["course_title"] for h in history[-3:]]

        context = f"""
EMPLOYEE PROFILE:
- ID: {employee['employee_id']}
- Department: {employee['department']}
- Role: {employee['job_title']} ({employee['job_level']})
- Location: {employee['location']}
- Years at AA: {employee['years_at_aa']}
- Performance: {employee['performance_rating']}
- Career Goal: {employee['career_goal']}
- Learning Hours YTD: {employee['learning_hours_ytd']}
- Preferred Format: {employee['preferred_format']}

CURRENT SKILLS: {', '.join(employee.get('current_skills', []))}
CERTIFICATIONS: {', '.join(employee.get('certifications', [])) or 'None yet'}

HIGH-PRIORITY SKILL GAPS: {', '.join(high_priority_gaps) or 'None identified'}
ALL SKILL GAPS: {len(gaps)} gaps identified

RECENTLY COMPLETED COURSES: {', '.join(recent_courses) or 'No recent completions'}
TOTAL COURSES COMPLETED: {len(history)}

PRE-FILTERED COURSE RECOMMENDATIONS:
{json.dumps([{"id": c["id"], "title": c["title"], "category": c["category"], 
              "duration": f"{c['duration_hours']}h", "format": c["format"], 
              "cost": "$" + str(c["cost"]) if c["cost"] > 0 else "Free"} 
             for c in recommended], indent=2)}
"""
        return context

    # ─── Agent Interaction ────────────────────────────────────────────────────

    def load_employee(self, employee_id: str) -> tuple[bool, str]:
        emp = self.get_employee(employee_id)
        if not emp:
            return False, f"Employee ID `{employee_id}` not found. Please check the ID and try again."

        self.current_employee = emp
        self.conversation_history = []  # Reset conversation for new employee

        intro = self.llm.chat(
            messages=[{
                "role": "user",
                "content": f"I just loaded an employee profile. Please give a brief, warm welcome and summary of this employee's learning status.\n\n{self._build_employee_context(emp)}"
            }],
            system_prompt=SYSTEM_PROMPT,
            max_tokens=400,
        )
        return True, intro

    def chat(self, user_message: str) -> str:
        if not self.current_employee:
            return "Please load an employee profile first by entering an Employee ID in the sidebar."

        context = self._build_employee_context(self.current_employee)

        # First message in session: inject full context
        if not self.conversation_history:
            first_message = {
                "role": "user",
                "content": f"Employee context for this session:\n{context}\n\nEmployee says: {user_message}"
            }
            self.conversation_history.append(first_message)
        else:
            self.conversation_history.append({"role": "user", "content": user_message})

        response = self.llm.chat(
            messages=self.conversation_history,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=800,
            temperature=0.7,
        )

        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def get_quick_analysis(self, analysis_type: str) -> str:
        """Generate specific analyses without conversation history."""
        if not self.current_employee:
            return "No employee loaded."

        context = self._build_employee_context(self.current_employee)

        prompts = {
            "recommendations": f"Based on this employee profile, provide a structured Learning & Development Plan with top 5 course recommendations, explaining WHY each is relevant.\n\n{context}",
            "skill_gaps": f"Analyze the skill gaps for this employee. Provide a prioritized gap analysis with specific actions to address each gap.\n\n{context}",
            "career_path": f"Suggest 2-3 realistic career progression paths for this employee at American Airlines, with the learning milestones needed for each path.\n\n{context}",
            "30_60_90": f"Create a 30-60-90 day learning plan for this employee that addresses their most critical development needs and career goals.\n\n{context}",
        }

        prompt = prompts.get(analysis_type, f"Provide insights on: {analysis_type}\n\n{context}")
        return self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=SYSTEM_PROMPT,
            max_tokens=900,
        )

    # ─── Analytics ────────────────────────────────────────────────────────────

    def get_dept_analytics(self) -> dict:
        employees_df = self.data["employees"]
        history_df = self.data["learning_history"]
        gaps_df = self.data["skill_gaps"]

        dept_stats = employees_df.groupby("department").agg(
            headcount=("employee_id", "count"),
            avg_learning_hours=("learning_hours_ytd", "mean"),
        ).reset_index()

        gap_counts = gaps_df.groupby("employee_id").size().reset_index(name="gap_count")
        emp_gaps = employees_df.merge(gap_counts, on="employee_id", how="left").fillna(0)
        dept_gaps = emp_gaps.groupby("department")["gap_count"].mean().reset_index(name="avg_gaps")
        dept_stats = dept_stats.merge(dept_gaps, on="department")

        return dept_stats.to_dict("records")

    def get_popular_courses(self) -> pd.DataFrame:
        history_df = self.data["learning_history"]
        popular = (
            history_df.groupby(["course_id", "course_title"])
            .agg(enrollments=("employee_id", "count"), avg_score=("score", "mean"), avg_rating=("feedback_rating", "mean"))
            .reset_index()
            .sort_values("enrollments", ascending=False)
            .head(10)
        )
        return popular

    def get_performance_learning_correlation(self) -> pd.DataFrame:
        employees_df = self.data["employees"]
        return employees_df.groupby("performance_rating").agg(
            avg_learning_hours=("learning_hours_ytd", "mean"),
            count=("employee_id", "count"),
        ).reset_index()