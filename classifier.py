"""
SUPERVISED: predict a resume's job category.
Pipeline: TF-IDF -> Logistic Regression.

Train (from project root):
    python -m src.models.classifier

Then import `predict_category` from the app to classify a new resume.
"""

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    CLASSIFIER_LABEL_ENCODER_PATH,
    CLASSIFIER_MODEL_PATH,
    CLASSIFIER_VECTORIZER_PATH,
    RANDOM_STATE,
    RESUME_CATEGORY_COL,
    RESUME_TEXT_COL,
    RESUMES_CLEAN_CSV,
)
from src.features.text_features import build_tfidf_vectorizer, transform_corpus


def train(csv_path=RESUMES_CLEAN_CSV, test_size: float = 0.2) -> dict:
    """Train the classifier and persist model + vectorizer + label encoder."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[RESUME_TEXT_COL, RESUME_CATEGORY_COL])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[RESUME_CATEGORY_COL])

    vectorizer = build_tfidf_vectorizer()
    X = vectorizer.fit_transform(df[RESUME_TEXT_COL].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
    )
    print(report)

    joblib.dump(model, CLASSIFIER_MODEL_PATH)
    joblib.dump(vectorizer, CLASSIFIER_VECTORIZER_PATH)
    joblib.dump(label_encoder, CLASSIFIER_LABEL_ENCODER_PATH)
    print(f"Saved model -> {CLASSIFIER_MODEL_PATH}")
    print(f"Saved vectorizer -> {CLASSIFIER_VECTORIZER_PATH}")
    print(f"Saved label encoder -> {CLASSIFIER_LABEL_ENCODER_PATH}")

    return {"report": report}


def load_artifacts():
    model = joblib.load(CLASSIFIER_MODEL_PATH)
    vectorizer = joblib.load(CLASSIFIER_VECTORIZER_PATH)
    label_encoder = joblib.load(CLASSIFIER_LABEL_ENCODER_PATH)
    return model, vectorizer, label_encoder


def predict_category(resume_text: str, top_k: int = 3) -> list[tuple[str, float]]:
    """
    Predict the top_k most likely categories for a single resume's raw text.
    Returns a list of (category, probability) sorted descending.
    """
    model, vectorizer, label_encoder = load_artifacts()
    X = transform_corpus([resume_text], vectorizer)
    probs = model.predict_proba(X)[0]
    ranked = sorted(zip(label_encoder.classes_, probs), key=lambda t: t[1], reverse=True)
    return ranked[:top_k]


if __name__ == "__main__":
    train()
