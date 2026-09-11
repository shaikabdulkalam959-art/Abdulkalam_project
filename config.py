"""
Central config for SmartHire.
Every path and column name used across src/ and app/ lives here so nothing
is hard-coded elsewhere. If your data/processed/*.csv columns differ from
the defaults below, change them here only.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths ---
ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (DATA_INTERIM, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RESUMES_CLEAN_CSV = DATA_PROCESSED / "resumes_clean.csv"
JOBS_CLEAN_CSV = DATA_PROCESSED / "jobs_clean.csv"
JOB_CORPUS_CSV = DATA_INTERIM / "job_corpus.csv"

# ------------------------------------------------------- model artifacts --
CLASSIFIER_MODEL_PATH = MODELS_DIR / "resume_classifier.pkl"
CLASSIFIER_VECTORIZER_PATH = MODELS_DIR / "resume_tfidf_vectorizer.pkl"
CLASSIFIER_LABEL_ENCODER_PATH = MODELS_DIR / "resume_label_encoder.pkl"

RECOMMENDER_VECTORIZER_PATH = MODELS_DIR / "job_tfidf_vectorizer.pkl"
RECOMMENDER_MATRIX_PATH = MODELS_DIR / "job_tfidf_matrix.pkl"

# ------------------------------------------------- resumes_clean.csv cols
RESUME_TEXT_COL = "Resume"          # cleaned resume text
RESUME_CATEGORY_COL = "Category"    # ground-truth label for the classifier

# ----------------------------------------------------- jobs_clean.csv cols
JOB_ID_COL = "job_id"
JOB_TITLE_COL = "Job Title"
JOB_SKILLS_COL = "Key Skills"           # raw skills string, comma/pipe separated
JOB_DESCRIPTION_COL = "Job Description"
JOB_EXPERIENCE_COL = "Job Experience Required"
JOB_ROLE_CATEGORY_COL = "Role Category"
JOB_LOCATION_COL = "Location"
# Combined free-text field the vectorizer is fit on (title + skills + description)
JOB_TEXT_COL = "job_text"

# ------------------------------------------------------------- modeling --
RANDOM_STATE = 42
TFIDF_MAX_FEATURES = 20_000
TFIDF_NGRAM_RANGE = (1, 2)
TOP_N_RECOMMENDATIONS = 10
