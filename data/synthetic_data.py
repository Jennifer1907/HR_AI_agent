"""
Synthetic HR Data Generator for American Airlines
Learning & Development Recommendation System
"""

import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

DEPARTMENTS = [
    "Flight Operations", "Cabin Crew Services", "Airport Operations",
    "Technical Operations", "Customer Experience", "Finance & Accounting",
    "Human Resources", "IT & Digital", "Safety & Compliance",
    "Revenue Management", "Cargo Operations", "Legal & Corporate Affairs"
]

JOB_ROLES = {
    "Flight Operations": ["Captain", "First Officer", "Flight Dispatcher", "Operations Coordinator"],
    "Cabin Crew Services": ["Flight Attendant", "Lead Cabin Crew", "Cabin Crew Supervisor", "Purser"],
    "Airport Operations": ["Gate Agent", "Ramp Agent", "Station Manager", "Ground Ops Supervisor"],
    "Technical Operations": ["Aircraft Mechanic", "Avionics Technician", "Quality Control Inspector", "Maintenance Manager"],
    "Customer Experience": ["Customer Service Agent", "Customer Relations Specialist", "CX Manager", "Loyalty Program Analyst"],
    "Finance & Accounting": ["Financial Analyst", "Senior Accountant", "Controller", "Budget Manager"],
    "Human Resources": ["HR Generalist", "Talent Acquisition Specialist", "Compensation Analyst", "HRBP"],
    "IT & Digital": ["Software Engineer", "Data Analyst", "Cybersecurity Specialist", "IT Manager"],
    "Safety & Compliance": ["Safety Officer", "Compliance Auditor", "Risk Analyst", "SMS Manager"],
    "Revenue Management": ["Revenue Analyst", "Pricing Specialist", "RM Manager", "Demand Planner"],
    "Cargo Operations": ["Cargo Agent", "Cargo Coordinator", "Dangerous Goods Specialist", "Cargo Manager"],
    "Legal & Corporate Affairs": ["Corporate Counsel", "Paralegal", "Contracts Manager", "Regulatory Affairs Specialist"]
}

SKILLS = {
    "Technical": [
        "Python Programming", "SQL", "Data Analysis", "Machine Learning",
        "Cloud Computing (AWS)", "Cybersecurity", "Aviation Systems", "SAP/ERP",
        "Aircraft Maintenance", "Avionics", "Navigation Systems", "Aircraft Performance"
    ],
    "Operations": [
        "Flight Operations", "Airport Ground Ops", "Safety Management Systems",
        "Crew Resource Management", "Emergency Procedures", "IATA Regulations",
        "FAA Compliance", "Dangerous Goods Handling", "Revenue Management", "Load Planning"
    ],
    "Leadership": [
        "Team Leadership", "Strategic Planning", "Change Management",
        "Performance Management", "Conflict Resolution", "Executive Communication",
        "Project Management", "Stakeholder Management", "Coaching & Mentoring"
    ],
    "Customer": [
        "Customer Service Excellence", "De-escalation Techniques", "Multilingual Communication",
        "Loyalty Program Management", "Complaint Resolution", "CRM Systems"
    ],
    "Business": [
        "Financial Analysis", "Budget Management", "Contract Negotiation",
        "Business Development", "Market Analysis", "Risk Management",
        "Process Improvement", "Six Sigma / Lean"
    ]
}

COURSES = [
    # Safety & Compliance
    {"id": "C001", "title": "FAA Part 121 Compliance Refresher", "category": "Safety & Compliance", "duration_hours": 8, "level": "Intermediate", "format": "In-Person", "provider": "AA Internal Training", "cost": 0},
    {"id": "C002", "title": "Crew Resource Management (CRM) Advanced", "category": "Safety & Compliance", "duration_hours": 16, "level": "Advanced", "format": "Simulator", "provider": "FlightSafety International", "cost": 1200},
    {"id": "C003", "title": "Dangerous Goods Handling Recurrent", "category": "Safety & Compliance", "duration_hours": 4, "level": "Foundational", "format": "Online", "provider": "IATA Training", "cost": 150},
    {"id": "C004", "title": "Safety Management Systems (SMS) Fundamentals", "category": "Safety & Compliance", "duration_hours": 6, "level": "Foundational", "format": "Online", "provider": "AA Internal Training", "cost": 0},
    
    # Leadership & Management
    {"id": "C005", "title": "Leadership Excellence Program", "category": "Leadership", "duration_hours": 40, "level": "Advanced", "format": "Blended", "provider": "Harvard Business Publishing", "cost": 2500},
    {"id": "C006", "title": "Frontline Manager Essentials", "category": "Leadership", "duration_hours": 24, "level": "Intermediate", "format": "Blended", "provider": "AA Leadership Academy", "cost": 0},
    {"id": "C007", "title": "Coaching for Performance", "category": "Leadership", "duration_hours": 12, "level": "Intermediate", "format": "Virtual Instructor-Led", "provider": "Blanchard International", "cost": 800},
    {"id": "C008", "title": "Strategic Thinking & Planning", "category": "Leadership", "duration_hours": 20, "level": "Advanced", "format": "Online", "provider": "LinkedIn Learning", "cost": 300},
    
    # Customer Service
    {"id": "C009", "title": "Delivering Exceptional Customer Experience", "category": "Customer Service", "duration_hours": 8, "level": "Foundational", "format": "Blended", "provider": "AA Customer Experience Team", "cost": 0},
    {"id": "C010", "title": "De-escalation & Conflict Resolution", "category": "Customer Service", "duration_hours": 6, "level": "Intermediate", "format": "In-Person", "provider": "AA Internal Training", "cost": 0},
    {"id": "C011", "title": "AAdvantage Loyalty Program Mastery", "category": "Customer Service", "duration_hours": 4, "level": "Foundational", "format": "Online", "provider": "AA Internal Training", "cost": 0},
    
    # Technology & Data
    {"id": "C012", "title": "Data Analytics with Python & SQL", "category": "Technology", "duration_hours": 32, "level": "Intermediate", "format": "Online", "provider": "Coursera (Google)", "cost": 400},
    {"id": "C013", "title": "Cloud Computing Fundamentals (AWS)", "category": "Technology", "duration_hours": 20, "level": "Intermediate", "format": "Online", "provider": "AWS Training", "cost": 300},
    {"id": "C014", "title": "Cybersecurity Awareness", "category": "Technology", "duration_hours": 2, "level": "Foundational", "format": "Online", "provider": "AA IT Security", "cost": 0},
    {"id": "C015", "title": "Machine Learning for Business", "category": "Technology", "duration_hours": 24, "level": "Advanced", "format": "Online", "provider": "Udacity", "cost": 600},
    
    # Finance & Business
    {"id": "C016", "title": "Aviation Revenue Management Fundamentals", "category": "Finance & Business", "duration_hours": 16, "level": "Intermediate", "format": "Virtual Instructor-Led", "provider": "IATA Training", "cost": 900},
    {"id": "C017", "title": "Financial Modeling & Forecasting", "category": "Finance & Business", "duration_hours": 20, "level": "Advanced", "format": "Online", "provider": "CFI Institute", "cost": 500},
    {"id": "C018", "title": "Project Management Professional (PMP) Prep", "category": "Business", "duration_hours": 36, "level": "Advanced", "format": "Online", "provider": "PMI", "cost": 700},
    {"id": "C019", "title": "Lean Six Sigma Green Belt", "category": "Business", "duration_hours": 40, "level": "Advanced", "format": "Blended", "provider": "ASQ", "cost": 1500},
    
    # HR & People
    {"id": "C020", "title": "HRBP Strategic Partnership", "category": "Human Resources", "duration_hours": 16, "level": "Advanced", "format": "Virtual Instructor-Led", "provider": "SHRM", "cost": 800},
    {"id": "C021", "title": "Talent Acquisition in Aviation", "category": "Human Resources", "duration_hours": 8, "level": "Intermediate", "format": "Online", "provider": "AA HR Academy", "cost": 0},
    {"id": "C022", "title": "Diversity, Equity & Inclusion in the Workplace", "category": "Human Resources", "duration_hours": 4, "level": "Foundational", "format": "Online", "provider": "AA Internal Training", "cost": 0},
    
    # Technical / Maintenance
    {"id": "C023", "title": "Boeing 737 MAX Systems Training", "category": "Technical", "duration_hours": 40, "level": "Advanced", "format": "Simulator", "provider": "Boeing Training", "cost": 3000},
    {"id": "C024", "title": "Airbus A321XLR Ground School", "category": "Technical", "duration_hours": 32, "level": "Advanced", "format": "In-Person", "provider": "Airbus Training", "cost": 2800},
    {"id": "C025", "title": "Aviation Maintenance Regulations (FAR Part 43)", "category": "Technical", "duration_hours": 8, "level": "Intermediate", "format": "Online", "provider": "FAA Safety", "cost": 0},
]

CERTIFICATIONS = [
    "FAA Airman Certificate", "IATA Dangerous Goods", "PMP", "Six Sigma Green Belt",
    "AWS Cloud Practitioner", "SHRM-CP", "SHRM-SCP", "CFA", "CPA",
    "CompTIA Security+", "Airline Transport Pilot (ATP)", "A&P Mechanic License"
]

PERFORMANCE_RATINGS = ["Exceeds Expectations", "Meets Expectations", "Needs Improvement", "Outstanding"]

CAREER_GOALS = [
    "Advance to management role", "Develop technical expertise",
    "Transition to different department", "Improve customer service skills",
    "Obtain industry certification", "Lead cross-functional projects",
    "Improve data & analytical skills", "Enhance safety knowledge"
]


# ─── GENERATORS ───────────────────────────────────────────────────────────────

def generate_employee(emp_id: int) -> dict:
    dept = random.choice(DEPARTMENTS)
    role = random.choice(JOB_ROLES[dept])
    years_at_aa = round(random.uniform(0.5, 25), 1)
    total_experience = round(years_at_aa + random.uniform(0, 15), 1)
    
    # Skill set
    all_skills = [s for cat in SKILLS.values() for s in cat]
    n_skills = random.randint(3, 10)
    employee_skills = random.sample(all_skills, n_skills)
    
    # Completed courses (subset)
    n_completed = random.randint(0, 8)
    completed_courses = random.sample([c["id"] for c in COURSES], min(n_completed, len(COURSES)))
    
    # Certifications
    n_certs = random.randint(0, 3)
    employee_certs = random.sample(CERTIFICATIONS, n_certs)
    
    hire_date = datetime.now() - timedelta(days=int(years_at_aa * 365))
    
    return {
        "employee_id": f"AA{emp_id:05d}",
        "name": f"Employee_{emp_id}",  # anonymized
        "department": dept,
        "job_title": role,
        "job_level": random.choice(["Individual Contributor", "Senior IC", "Team Lead", "Manager", "Director"]),
        "location": random.choice(["DFW (HQ)", "CLT", "MIA", "LAX", "JFK", "ORD", "PHX", "PHL"]),
        "hire_date": hire_date.strftime("%Y-%m-%d"),
        "years_at_aa": years_at_aa,
        "total_experience_years": total_experience,
        "performance_rating": random.choice(PERFORMANCE_RATINGS),
        "current_skills": employee_skills,
        "certifications": employee_certs,
        "completed_courses": completed_courses,
        "career_goal": random.choice(CAREER_GOALS),
        "learning_hours_ytd": round(random.uniform(0, 80), 1),
        "preferred_format": random.choice(["Online", "In-Person", "Blended", "Virtual Instructor-Led"]),
        "manager_id": f"AA{random.randint(1, 50):05d}" if emp_id > 50 else None,
        "team_size": random.randint(0, 15) if "Manager" in role or "Supervisor" in role or "Lead" in role else 0,
        "salary_band": random.choice(["Band 1", "Band 2", "Band 3", "Band 4", "Band 5"]),
    }


def generate_learning_history(employees: list) -> list:
    history = []
    for emp in employees:
        for course_id in emp["completed_courses"]:
            course = next(c for c in COURSES if c["id"] == course_id)
            completion_date = datetime.now() - timedelta(days=random.randint(30, 730))
            history.append({
                "employee_id": emp["employee_id"],
                "course_id": course_id,
                "course_title": course["title"],
                "completion_date": completion_date.strftime("%Y-%m-%d"),
                "score": round(random.uniform(70, 100), 1),
                "time_spent_hours": round(course["duration_hours"] * random.uniform(0.8, 1.3), 1),
                "feedback_rating": random.randint(3, 5),
                "feedback_text": random.choice([
                    "Very relevant to my daily work",
                    "Could use more hands-on exercises",
                    "Excellent content, well-structured",
                    "Good overview but needs more depth",
                    "Highly recommend to colleagues",
                    "Met expectations",
                    "Transformative learning experience"
                ])
            })
    return history


def generate_skill_gaps(employees: list) -> list:
    gaps = []
    for emp in employees:
        dept_roles = JOB_ROLES[emp["department"]]
        required_skills_pool = []
        
        # Simulate required skills per role
        if emp["department"] in ["Safety & Compliance", "Flight Operations", "Technical Operations"]:
            required_skills_pool = SKILLS["Technical"] + SKILLS["Operations"]
        elif emp["department"] in ["IT & Digital"]:
            required_skills_pool = SKILLS["Technical"] + SKILLS["Business"]
        elif emp["department"] in ["Customer Experience", "Cabin Crew Services", "Airport Operations"]:
            required_skills_pool = SKILLS["Customer"] + SKILLS["Operations"]
        elif emp["department"] in ["Finance & Accounting", "Revenue Management"]:
            required_skills_pool = SKILLS["Business"] + SKILLS["Technical"][:4]
        else:
            required_skills_pool = SKILLS["Leadership"] + SKILLS["Business"]
        
        required = random.sample(required_skills_pool, min(6, len(required_skills_pool)))
        missing = [s for s in required if s not in emp["current_skills"]]
        
        for skill in missing:
            gaps.append({
                "employee_id": emp["employee_id"],
                "skill": skill,
                "current_proficiency": random.choice(["None", "Basic", "Developing"]),
                "required_proficiency": random.choice(["Intermediate", "Advanced"]),
                "priority": random.choice(["High", "Medium", "Low"]),
                "identified_date": (datetime.now() - timedelta(days=random.randint(0, 180))).strftime("%Y-%m-%d")
            })
    return gaps


def build_datasets(n_employees: int = 200) -> dict:
    employees = [generate_employee(i) for i in range(1, n_employees + 1)]
    learning_history = generate_learning_history(employees)
    skill_gaps = generate_skill_gaps(employees)
    
    return {
        "employees": employees,
        "courses": COURSES,
        "learning_history": learning_history,
        "skill_gaps": skill_gaps,
    }


def get_dataframes(n_employees: int = 200):
    data = build_datasets(n_employees)
    return {
        "employees": pd.DataFrame(data["employees"]),
        "courses": pd.DataFrame(data["courses"]),
        "learning_history": pd.DataFrame(data["learning_history"]),
        "skill_gaps": pd.DataFrame(data["skill_gaps"]),
    }


if __name__ == "__main__":
    dfs = get_dataframes(200)
    for name, df in dfs.items():
        print(f"{name}: {df.shape}")
        print(df.head(2))
        print()