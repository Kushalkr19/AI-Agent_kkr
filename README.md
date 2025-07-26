# AI Agent Resume Matcher

This repository contains an AI-powered resume matcher that helps you tailor your resume to a specific job description.

## Overview

The `resume_matcher.py` script compares the content of a résumé and a job description using TF‑IDF vectorization and cosine similarity. It extracts the most frequent keywords from the job description, checks which appear in the résumé, and orders bullet points by relevance. The tool runs locally and doesn’t send any data to external services.

## Usage

1. Ensure you have Python 3 and `scikit‑learn` installed.
2. Place your résumé (PDF or `.txt`) and job description (`.pdf` or `.txt`) in the same folder.
3. Run the script from the command line:

```bash
python resume_matcher.py your_resume.pdf job_description.txt
```

4. The script will generate `match_report.txt` and `curated_resume.txt` with detailed feedback.

See the script’s docstring for more details.

## Notes

- The match score is a simple measure of textual overlap; use it as a guide rather than a definitive ranking.
- Only include keywords in your résumé if they genuinely represent your experience.
