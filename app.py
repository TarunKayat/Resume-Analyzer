import streamlit as st
import fitz
import numpy as np
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import pandas as pd

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    ListFlowable, ListItem
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# ================= ENV =================
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
ai_model = genai.GenerativeModel("gemini-flash-latest")

# ================= FUNCTIONS =================

def extract_text(pdf):
    text = ""
    doc = fitz.open(stream=pdf.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text


def match_resume(resume, job):
    emb = embed_model.encode([resume, job])
    score = np.dot(emb[0], emb[1]) / (
        np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])
    )
    return round(score * 100, 2)


skill_db = [
    "python","machine learning","sql","nlp","tensorflow",
    "excel","data science","flask","streamlit","git"
]

def extract_skills(text):
    return [s for s in skill_db if s in text.lower()]


def ai_feedback(resume, job):
    prompt = f"""
You are an ATS system.

Resume:
{resume}

Job Description:
{job}

Return:
- ATS score
- Missing skills
- Resume improvement tips in bullet points
"""
    return ai_model.generate_content(prompt).text


def resume_chatbot(resume, question):
    prompt = f"""
Resume:
{resume}

Question:
{question}

Answer clearly.
"""
    return ai_model.generate_content(prompt).text


def mock_interview(resume):
    prompt = f"""
Based on this resume generate:
5 technical interview questions
3 HR questions
2 project deep dive questions
"""
    return ai_model.generate_content(prompt).text


# ================= PDF REPORT =================

def create_pdf(score, found, missing, feedback):
    file = "ATS_Report.pdf"
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(file)

    elements = []

    elements.append(Paragraph("AI Resume ATS Report", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"<b>ATS Score:</b> {score}%", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Skills Found", styles["Heading2"]))
    elements.append(ListFlowable(
        [ListItem(Paragraph(s, styles["Normal"])) for s in found],
        bulletType="bullet"
    ))

    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("Missing Skills", styles["Heading2"]))
    elements.append(ListFlowable(
        [ListItem(Paragraph(s, styles["Normal"])) for s in missing],
        bulletType="bullet"
    ))

    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph("AI Feedback & Tips", styles["Heading2"]))

    feedback_lines = [l for l in feedback.split("\n") if l.strip()]

    elements.append(ListFlowable(
        [ListItem(Paragraph(l, styles["Normal"])) for l in feedback_lines],
        bulletType="bullet"
    ))

    doc.build(elements)
    return file


# ================= LEADERBOARD =================

def update_board(name, score):
    file = "leaderboard.csv"

    if os.path.exists(file):
        df = pd.read_csv(file)
    else:
        df = pd.DataFrame(columns=["Name", "Score"])

    df.loc[len(df)] = [name, score]
    df = df.sort_values(by="Score", ascending=False)
    df.to_csv(file, index=False)

    return df


# ================= UI =================

st.set_page_config("AI Resume ATS", "📄")
st.title("📄 AI Resume Analyzer Platform")

resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
job_desc = st.text_area("Paste Job Description")

if resume_file and job_desc:

    with st.spinner("Analyzing with AI..."):
        resume_text = extract_text(resume_file)
        score = match_resume(resume_text, job_desc)
        found = extract_skills(resume_text)
        missing = [s for s in skill_db if s not in found]
        feedback = ai_feedback(resume_text, job_desc)

    st.subheader("📊 ATS Score")
    st.progress(int(score))
    st.write(f"{score}%")

    st.subheader("✅ Skills Found")
    st.write(found)

    st.subheader("❌ Skill Gaps")
    st.write(missing)

    st.subheader("🤖 AI Feedback")
    st.write(feedback)

    # ---- Chatbot ----
    st.subheader("💬 Resume Chatbot")
    q = st.text_input("Ask about your resume")
    if q:
        st.write(resume_chatbot(resume_text, q))

    # ---- Mock Interview ----
    st.subheader("🎥 AI Mock Interview")
    if st.button("Generate Interview Questions"):
        st.write(mock_interview(resume_text))

    # ---- PDF ----
    if st.button("📄 Download ATS Report"):
        pdf = create_pdf(score, found, missing, feedback)
        with open(pdf, "rb") as f:
            st.download_button("Download PDF", f, file_name="ATS_Report.pdf")

    # ---- Leaderboard ----
    st.subheader("🏆 Resume Leaderboard")
    name = st.text_input("Enter your name")

    if st.button("Submit Score"):
        board = update_board(name, score)
        st.dataframe(board.head(10))
