"""
Feature helpers for comparing a resume against a job posting:
skill overlap and the skill-gap report (job skills minus resume skills).
"""

import re
from typing import Iterable

_SPLIT_RE = re.compile(r"[,/|;•\n]")


def split_skills(raw: str) -> set[str]:
    """Turn a raw 'Key Skills' or free-text field into a normalized skill set."""
    if not isinstance(raw, str) or not raw.strip():
        return set()
    parts = _SPLIT_RE.split(raw)
    skills = {p.strip().lower() for p in parts if p.strip()}
    return skills


def extract_resume_skills(resume_text: str, known_skills: Iterable[str]) -> set[str]:
    """
    Match a resume's free text against a vocabulary of known skills
    (e.g. the union of every skill seen in jobs_clean.csv) via substring search.
    """
    text = resume_text.lower() if isinstance(resume_text, str) else ""
    found = set()
    for skill in known_skills:
        skill_norm = skill.strip().lower()
        if not skill_norm:
            continue
        # word-boundary-ish match so "r" doesn't match every word containing r
        pattern = r"(?<![a-zA-Z])" + re.escape(skill_norm) + r"(?![a-zA-Z])"
        if re.search(pattern, text):
            found.add(skill_norm)
    return found


def skill_overlap_score(resume_skills: set[str], job_skills: set[str]) -> float:
    """Fraction of the job's required skills the resume already covers (0-1)."""
    if not job_skills:
        return 0.0
    return len(resume_skills & job_skills) / len(job_skills)


def skill_gap(resume_skills: set[str], job_skills: set[str]) -> dict:
    """
    Skill-gap report: what the job wants that the resume doesn't show,
    what the resume already covers, and the coverage score.
    """
    matched = sorted(resume_skills & job_skills)
    missing = sorted(job_skills - resume_skills)
    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "coverage": skill_overlap_score(resume_skills, job_skills),
    }
