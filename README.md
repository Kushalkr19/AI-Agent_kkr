# AI Agent Résumé Matcher

This repository contains an AI‑powered résumé matcher designed to help you
tailor your résumé to a specific job description.  It builds upon the
original ``resume_matcher.py`` script by incorporating contextual matching
alongside keyword overlap and by producing a ranked outline of your résumé
with suggestions for incorporating missing keywords.

## Overview

The core logic lives in ``resume_agent.py``.  Given a résumé and a job
description (PDF or plain‑text), it:

1. Extracts text from the documents using the ``pdftotext`` utility for PDFs.
2. Normalises and tokenises the text to remove punctuation and stop words.
3. Computes both an overall similarity score between the résumé and the job
   description, and per‑bullet relevance scores using TF–IDF and cosine
   similarity to estimate contextual alignment.
4. Extracts the most frequent keywords from the job description and
   determines which are present or missing in the résumé.
5. Ranks résumé bullet points by a weighted combination of keyword and
   contextual relevance.
6. Generates a match report summarising the scores and a curated résumé
   outline with suggestions for incorporating missing keywords.

Run the script with:

```bash
python resume_agent.py your_resume.pdf job_description.txt
```

This will create ``match_report.txt`` and ``curated_resume.txt`` in the
current directory.  See the docstring at the top of ``resume_agent.py`` for
more details and additional command‑line options.

## Compatibility Note

For convenience the legacy entry point ``resume_matcher.py`` now simply
invokes the enhanced agent.  You can still run:

```bash
python resume_matcher.py your_resume.pdf job_description.txt
```

and it will delegate to ``resume_agent.py``.