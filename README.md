I built out the three stub pieces your README calls out — resume classifier, job recommender, and the Streamlit app — plus supporting feature/parsing code:
classifier.py — TF-IDF → Logistic Regression, predicts resume category with confidence scores
recommender.py — TF-IDF + cosine similarity, returns top-N matching jobs
match_features.py — skill overlap + skill-gap report (job skills minus resume skills)
resume_parser.py — extracts text from PDF/DOCX/TXT uploads
streamlit_app.py — wires all of it into the upload → classify → recommend → skill-gap flow
config.py, text_features.py, evaluate.py — shared plumbing
