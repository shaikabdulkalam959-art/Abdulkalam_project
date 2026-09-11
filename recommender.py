"""
UNSUPERVISED: recommend the top-N most similar jobs to a resume.
Pipeline: TF-IDF over the job corpus -> cosine similarity against the
resume's TF-IDF vector.

Fit (from project root):
    python -m src.models.recommender

Then import `recommend_jobs` from the app to rank jobs for a new resume.
"""

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    JOB_DESCRIPTION_COL,
    JOB_SKILLS_COL,
    JOB_TEXT_COL,
    JOB_TITLE_COL,
    JOBS_CLEAN_CSV,
    RECOMMENDER_MATRIX_PATH,
    RECOMMENDER_VECTORIZER_PATH,
    TOP_N_RECOMMENDATIONS,
)
from src.features.text_features import build_tfidf_vectorizer, transform_corpus


def _combine_job_text(row: pd.Series) -> str:
    """Build one free-text field per job from title + skills + description."""
    parts = [
        str(row.get(JOB_TITLE_COL, "")),
        str(row.get(JOB_SKILLS_COL, "")),
        str(row.get(JOB_DESCRIPTION_COL, "")),
    ]
    return " ".join(p for p in parts if p and p.lower() != "nan")


def fit(csv_path=JOBS_CLEAN_CSV) -> None:
    """Fit the job TF-IDF matrix and persist it + the vectorizer + the job table."""
    df = pd.read_csv(csv_path)
    df[JOB_TEXT_COL] = df.apply(_combine_job_text, axis=1)

    vectorizer = build_tfidf_vectorizer()
    matrix = vectorizer.fit_transform(df[JOB_TEXT_COL])

    joblib.dump(vectorizer, RECOMMENDER_VECTORIZER_PATH)
    joblib.dump({"matrix": matrix, "jobs": df}, RECOMMENDER_MATRIX_PATH)
    print(f"Saved job vectorizer -> {RECOMMENDER_VECTORIZER_PATH}")
    print(f"Saved job matrix + table -> {RECOMMENDER_MATRIX_PATH} ({len(df)} jobs)")


def load_artifacts():
    vectorizer = joblib.load(RECOMMENDER_VECTORIZER_PATH)
    payload = joblib.load(RECOMMENDER_MATRIX_PATH)
    return vectorizer, payload["matrix"], payload["jobs"]


def recommend_jobs(resume_text: str, top_n: int = TOP_N_RECOMMENDATIONS) -> pd.DataFrame:
    """
    Rank every job in the corpus by cosine similarity to the given resume text.
    Returns the top_n rows of the job table with an added `match_score` column,
    sorted descending.
    """
    vectorizer, job_matrix, jobs_df = load_artifacts()
    resume_vec = transform_corpus([resume_text], vectorizer)

    scores = cosine_similarity(resume_vec, job_matrix).flatten()
    result = jobs_df.copy()
    result["match_score"] = scores
    result = result.sort_values("match_score", ascending=False).head(top_n)
    return result.reset_index(drop=True)


if __name__ == "__main__":
    fit()
