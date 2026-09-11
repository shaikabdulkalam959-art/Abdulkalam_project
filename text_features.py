"""
Shared text-cleaning and TF-IDF vectorization utilities.
Used by both the resume classifier (src/models/classifier.py) and the
job recommender (src/models/recommender.py) so the two stay consistent.
"""

import re

from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE

_URL_RE = re.compile(r"http\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/emails/punctuation/digits, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def build_tfidf_vectorizer(**overrides) -> TfidfVectorizer:
    """Factory so classifier and recommender always build a TF-IDF the same way."""
    params = dict(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words="english",
        sublinear_tf=True,
    )
    params.update(overrides)
    return TfidfVectorizer(**params)


def fit_transform_corpus(texts, vectorizer: TfidfVectorizer = None):
    """Clean a list/Series of raw texts and fit+transform a TF-IDF vectorizer."""
    if vectorizer is None:
        vectorizer = build_tfidf_vectorizer()
    cleaned = [clean_text(t) for t in texts]
    matrix = vectorizer.fit_transform(cleaned)
    return matrix, vectorizer


def transform_corpus(texts, vectorizer: TfidfVectorizer):
    """Clean text and transform with an already-fit vectorizer."""
    cleaned = [clean_text(t) for t in texts]
    return vectorizer.transform(cleaned)
