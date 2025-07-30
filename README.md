# AI Agent Resume Matcher

This repository contains an AI‑powered résumé matcher designed to help you
tailor your resume to a specific job description.  It builds upon the
original ``resume_matcher.py`` script by incorporating contextual matching
alongside keyword overlap and by producing a ranked outline of your resume
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

### Advanced features

The latest version adds optional support for **semantic similarity** and
**LLM‑based bullet rewriting**.  These features are disabled by default and
require extra Python packages and (for rewriting) an API key:

* **Semantic search** (``--semantic``): uses
  [SentenceTransformers](https://www.sbert.net) to embed sentences and
  [FAISS](https://github.com/facebookresearch/faiss) to compute similarity.  This
  provides richer matching than TF–IDF by capturing the meaning of sentences.
  To enable it, install ``sentence-transformers`` and ``faiss`` and run the
  script with ``--semantic``.  If the libraries are missing, the agent falls
  back to TF–IDF automatically.

* **LLM rewriting** (``--rewrite``): rewrites bullet points using a large
  language model such as GPT‑4.  This can inject missing keywords
  and polish your achievements for better alignment with the job description.
  To enable rewriting, install the ``openai`` Python package, set the
  ``OPENAI_API_KEY`` environment variable, and run the script with
  ``--rewrite``.  Without an API key the bullets are left unchanged.

These advanced layers mirror the conceptual pipeline of a SaaS tool:

| Layer                    | Tools Used                         | Purpose                                          |
| ------------------------ | ---------------------------------- | ------------------------------------------------ |
| **LLM Rewriting Engine** | OpenAI GPT‑4                       | Rewrite and inject missing keywords contextually |
| **Semantic Matching**    | SentenceTransformers + FAISS        | Bullet <-> JD matching beyond exact words        |
| **Keyword Extraction**   | TF‑IDF + simple NER/stop‑word removal | Highlight must‑have keywords for ATS             |
| **Experience Generator** | Prompt chaining + user’s projects    | Add realistic projects aligned with the JD       |

Note: The repository does not bundle these heavy models; you must install
them separately if you wish to use the advanced options.

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
