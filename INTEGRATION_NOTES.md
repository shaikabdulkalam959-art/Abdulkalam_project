# How to use these files

Drop each file into the matching path in your `SmartHire` repo (they overwrite
the stub files with the same names):

```
src/config.py
src/features/text_features.py
src/features/match_features.py
src/models/classifier.py
src/models/recommender.py
src/parsing/resume_parser.py
src/evaluate.py
app/streamlit_app.py
```

## Assumptions I made (fix if yours differ)

I don't have access to your actual `data/processed/*.csv` files, so I built
against the standard columns for the datasets your `download_data.py`
references:

- **resumes_clean.csv** → `Category`, `Resume` (from
  `jillanisofttech/updated-resume-dataset`)
- **jobs_clean.csv** → `Job Title`, `Key Skills`, `Job Description`,
  `Job Experience Required`, `Role Category`, `Location` (from
  `PromptCloudHQ/jobs-on-naukricom`)

All of these are constants at the top of `src/config.py`
(`RESUME_TEXT_COL`, `JOB_SKILLS_COL`, etc.) — if your `preprocess.py`
produced different column names, change them there once and everything
else (classifier, recommender, app) picks it up automatically.

## New dependencies to add to `requirements.txt`

```
scikit-learn
pandas
joblib
streamlit
matplotlib
pypdf
python-docx
```

## Order to run things

```bash
# 1) train the classifier (saves models/resume_classifier.pkl etc.)
python -m src.models.classifier

# 2) fit the recommender (saves models/job_tfidf_matrix.pkl etc.)
python -m src.models.recommender

# 3) launch the app
streamlit run app/streamlit_app.py
```

## What's still up to you

- `04_clustering_topics.ipynb` (clustering/topics) and `05_fit_predictor.ipynb`
  (optional shortlisting model) are marked optional in your README — I left
  `src/models/clustering.py` and `src/models/fit_predictor.py` untouched
  since they're not part of the core scope.
- If `Job Experience Required`/`Role Category` aren't in your cleaned CSV,
  the app just won't display them — nothing breaks, since I only pull
  columns that exist.
