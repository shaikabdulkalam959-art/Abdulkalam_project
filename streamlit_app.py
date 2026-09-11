"""SmartHire web portal (Streamlit). Run: streamlit run app/streamlit_app.py"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow `from src...` imports when Streamlit runs this file directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    CLASSIFIER_MODEL_PATH,
    JOB_SKILLS_COL,
    JOB_TITLE_COL,
    RECOMMENDER_MATRIX_PATH,
)
from src.features.match_features import extract_resume_skills, skill_gap, split_skills
from src.models.classifier import predict_category
from src.models.recommender import recommend_jobs
from src.parsing.resume_parser import parse_resume

st.set_page_config(page_title="SmartHire", page_icon="🧭", layout="wide")

st.title("🧭 SmartHire")
st.caption("Upload a resume to get matching jobs, a predicted role category, and a skill-gap report.")

MODELS_READY = CLASSIFIER_MODEL_PATH.exists() and RECOMMENDER_MATRIX_PATH.exists()

if not MODELS_READY:
    st.warning(
        "Models haven't been trained yet. From the project root, run:\n\n"
        "```\npython -m src.models.classifier\npython -m src.models.recommender\n```"
    )
    st.stop()

uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    with st.spinner("Reading resume..."):
        try:
            resume_text = parse_resume(uploaded_file, filename=uploaded_file.name)
        except ValueError as e:
            st.error(str(e))
            st.stop()

    if not resume_text.strip():
        st.error("Couldn't extract any text from that file. Try a different format.")
        st.stop()

    with st.expander("Extracted resume text"):
        st.text(resume_text[:3000] + ("..." if len(resume_text) > 3000 else ""))

    col1, col2 = st.columns([1, 2])

    # --- Predicted category -------------------------------------------------
    with col1:
        st.subheader("Predicted category")
        with st.spinner("Classifying..."):
            top_categories = predict_category(resume_text, top_k=3)
        for category, prob in top_categories:
            st.metric(category, f"{prob:.0%}")

    # --- Matching jobs --------------------------------------------------------
    with col2:
        st.subheader("Top matching jobs")
        with st.spinner("Ranking jobs..."):
            top_jobs = recommend_jobs(resume_text, top_n=10)

        display_cols = [c for c in [JOB_TITLE_COL, JOB_SKILLS_COL, "match_score"] if c in top_jobs.columns]
        st.dataframe(
            top_jobs[display_cols].style.format({"match_score": "{:.2f}"}),
            use_container_width=True,
        )

    # --- Skill-gap report -------------------------------------------------
    st.subheader("Skill-gap report")
    job_choices = top_jobs[JOB_TITLE_COL].tolist() if JOB_TITLE_COL in top_jobs.columns else []
    if job_choices:
        selected_title = st.selectbox("Compare against which job?", job_choices)
        selected_row = top_jobs[top_jobs[JOB_TITLE_COL] == selected_title].iloc[0]

        job_skills = split_skills(selected_row.get(JOB_SKILLS_COL, ""))
        # Build a known-skill vocabulary from every job's skills column so we can
        # find which of those skills actually appear in the resume text.
        all_job_skills = set()
        if JOB_SKILLS_COL in top_jobs.columns:
            for raw in top_jobs[JOB_SKILLS_COL]:
                all_job_skills |= split_skills(raw)

        resume_skills = extract_resume_skills(resume_text, all_job_skills)
        report = skill_gap(resume_skills, job_skills)

        gcol1, gcol2, gcol3 = st.columns(3)
        gcol1.metric("Coverage", f"{report['coverage']:.0%}")
        gcol2.metric("Matched skills", len(report["matched_skills"]))
        gcol3.metric("Missing skills", len(report["missing_skills"]))

        st.markdown("**✅ Skills you already show:**")
        st.write(", ".join(report["matched_skills"]) or "—")

        st.markdown("**📌 Skills this job wants that your resume doesn't show:**")
        st.write(", ".join(report["missing_skills"]) or "—")
    else:
        st.info("No job skills data available to build a skill-gap report.")
else:
    st.info("👆 Upload a PDF, DOCX, or TXT resume to get started.")
