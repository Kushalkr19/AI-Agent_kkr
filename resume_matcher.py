#!/usr/bin/env python3
"""
resume_matcher.py
===================

This script implements a simple resume matcher that compares the content of a
candidate's résumé with a job description and generates a match report.  It
operates entirely on your local machine and does **not** send any data to
external services.  The matcher uses basic natural language processing (NLP)
techniques (tokenisation, stop‑word removal and TF‑IDF vectorisation) to
measure similarity between the résumé and the job description and to surface
important keywords.

Key features
------------

* Accepts résumés and job descriptions as PDF or plain‑text (``.txt``) files.
* Extracts text from PDF using the command‑line utility ``pdftotext`` (part
  of the poppler suite, which is available by default on most Linux
  distributions, including this environment).  For plain text files the
  contents are read directly.
* Preprocesses text by lowercasing, removing punctuation and a list of
  English stop‑words, and splitting into tokens.
* Computes a TF‑IDF representation of the résumé and the job description
  using ``scikit‑learn`` and measures the cosine similarity between them.
* Extracts the most frequent terms in the job description to identify
  important keywords, and determines how many of these keywords appear in
  the résumé.
* Heuristically identifies bullet points in the résumé (lines starting with
  common bullet characters like ``•``, ``-`` or ``*``) and sorts them
  based on how many job keywords they contain.  This allows the candidate to
  prioritise the most relevant achievements.
* Generates a curated résumé outline with the bullet points sorted by
  relevance and a summary of missing keywords that the candidate may wish to
  address.

Usage
-----

This script is intended to be run from the command line.  To see the
available options, run it with the ``-h`` flag:

```
python resume_matcher.py -h
```

The simplest invocation requires two positional arguments: the path to the
résumé and the path to the job description.  For example:

```
python resume_matcher.py my_resume.pdf job_description.txt
```

By default the script prints a short report to the console and writes two
files into the current working directory:

* ``match_report.txt`` – a textual summary of the similarity score,
  keyword coverage and a list of missing keywords.
* ``curated_resume.txt`` – the résumé bullet points sorted by relevance
  followed by any remaining content in its original order.

You can specify alternative output paths with the ``--report`` and
``--curated`` options.

Limitations
-----------

This tool uses simple NLP techniques and does **not** attempt to infer the
semantics of your experience beyond keyword overlap.  It is designed to help
you understand which parts of your résumé are most aligned with a given job
description and which important terms might be missing.  It is **not** a
replacement for human judgement or a guarantee of success.  Always ensure
that any suggested additions accurately reflect your skills and experience.
"""

import argparse
import os
import re
import string
import subprocess
from collections import Counter
from typing import List, Tuple, Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Define a basic list of English stop words.  Because external downloads are
# disabled in this environment we cannot fetch the NLTK stop‑words corpus
# dynamically, so this list is assembled manually.  It includes many common
# function words and pronouns.  Feel free to extend it if needed.
STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an',
    'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been',
    'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't",
    'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't",
    'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further',
    'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he',
    "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself',
    'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm",
    "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its',
    'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself',
    'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other',
    'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same',
    "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't",
    'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs',
    'them', 'themselves', 'then', 'there', "there's", 'these', 'they',
    "they'd", "they'll", "they're", "they've", 'this', 'those', 'through',
    'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we',
    "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's",
    'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's",
    'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you',
    "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself',
    'yourselves'
}


def extract_text(path: str) -> str:
    """Extract plain text from a PDF or text file.

    If the file extension is ``.pdf`` this function will call the external
    command ``pdftotext`` to convert the PDF into plain UTF‑8 text.
    Otherwise the file is assumed to be a UTF‑8 encoded text file and is
    read directly.  Unsupported extensions will raise a ``ValueError``.

    Parameters
    ----------
    path : str
        Path to the résumé or job description file.

    Returns
    -------
    str
        The extracted text.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf':
        # Use pdftotext to convert to text.  '-q' suppresses stderr unless
        # there are errors, '-' makes it write to stdout.
        try:
            output = subprocess.check_output([
                'pdftotext', '-q', path, '-'  # output to stdout
            ])
        except FileNotFoundError:
            raise RuntimeError(
                'pdftotext is required to extract text from PDF files. '
                'Please install the poppler utilities.'
            )
        return output.decode('utf-8', errors='ignore')
    elif ext == '.txt':
        with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
            return fh.read()
    else:
        raise ValueError(f'Unsupported file type: {ext}')


def preprocess(text: str) -> List[str]:
    """Preprocess a text string into a list of tokens.

    This function lowercases the text, removes punctuation, splits it on
    whitespace and filters out stop words and numeric tokens.

    Parameters
    ----------
    text : str
        Raw text to be processed.

    Returns
    -------
    List[str]
        A list of cleaned, tokenised words.
    """
    # Lowercase
    text = text.lower()
    # Replace punctuation with spaces
    trans_table = str.maketrans({k: ' ' for k in string.punctuation})
    text = text.translate(trans_table)
    # Split into tokens
    tokens = text.split()
    # Filter out stop words and single‑character tokens (often noise)
    cleaned = [t for t in tokens if t not in STOP_WORDS and len(t) > 1 and not t.isdigit()]
    return cleaned


def extract_bullets(text: str) -> List[str]:
    """Extract bullet lines from a résumé.

    This function splits the résumé into lines and returns those lines that
    appear to be bullet points, identified by starting with common bullet
    characters (``•``, ``-`` or ``*``) possibly preceded by whitespace or
    numbering.  The bullet character and any leading whitespace are stripped
    from the returned lines.

    Parameters
    ----------
    text : str
        The raw résumé text.

    Returns
    -------
    List[str]
        A list of bullet point strings.
    """
    bullet_lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        # Regex: optional digits and punctuation (for numbered lists), followed by a bullet char
        if re.match(r'^(\d+\.\s*)?[\u2022\-\*]+\s+', stripped):
            # Remove the bullet marker and leading/trailing whitespace
            cleaned = re.sub(r'^(\d+\.\s*)?[\u2022\-\*]+\s+', '', stripped).strip()
            if cleaned:
                bullet_lines.append(cleaned)
    return bullet_lines


def compute_similarity(resume_text: str, job_text: str) -> float:
    """Compute the cosine similarity between the résumé and job description.

    This function uses ``TfidfVectorizer`` to build a vocabulary across both
    documents and then computes the cosine similarity of the resulting
    TF‑IDF vectors.  The similarity score will be a float between 0 and 1.

    Parameters
    ----------
    resume_text : str
        Text extracted from the résumé.
    job_text : str
        Text extracted from the job description.

    Returns
    -------
    float
        The cosine similarity score between the two documents.
    """
    # We do not specify stop_words here because the texts passed into this
    # function should already have been preprocessed (tokenised and cleaned).
    vectoriser = TfidfVectorizer()
    tfidf_matrix = vectoriser.fit_transform([resume_text, job_text])
    sim_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return float(sim_matrix[0, 0])


def get_top_keywords(tokens: List[str], n: int = 20) -> List[str]:
    """Return the top ``n`` most common tokens in a list.

    Parameters
    ----------
    tokens : List[str]
        A list of preprocessed tokens.
    n : int, optional
        The number of keywords to return (default is 20).

    Returns
    -------
    List[str]
        A list of the most frequent tokens, sorted by frequency.
    """
    counter = Counter(tokens)
    return [word for word, _ in counter.most_common(n)]


def sort_bullets_by_relevance(bullets: List[str], keywords: Iterable[str]) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Sort résumé bullet points by how many job keywords they contain.

    Parameters
    ----------
    bullets : List[str]
        The list of résumé bullet points.
    keywords : Iterable[str]
        An iterable of job description keywords.

    Returns
    -------
    Tuple[List[Tuple[str, int]], List[str]]
        Two lists: the first contains tuples of (bullet, score) for bullets
        containing one or more keywords, sorted from highest to lowest score;
        the second contains any remaining bullets (with score zero) in their
        original order.
    """
    keyword_set = set(keywords)
    relevant = []
    irrelevant = []
    for bullet in bullets:
        bullet_tokens = preprocess(bullet)
        score = sum(1 for token in bullet_tokens if token in keyword_set)
        if score > 0:
            relevant.append((bullet, score))
        else:
            irrelevant.append(bullet)
    # Sort relevant bullets by score (descending) and preserve their original order if scores tie
    relevant_sorted = sorted(relevant, key=lambda x: (-x[1], bullets.index(x[0])))
    return relevant_sorted, irrelevant


def write_report(similarity: float,
                 job_keywords: List[str],
                 resume_tokens: List[str],
                 report_path: str) -> None:
    """Write the match report to a file.

    The report includes the similarity score, the percentage of job keywords
    present in the résumé, the list of job keywords and which ones are missing.

    Parameters
    ----------
    similarity : float
        The cosine similarity score between the résumé and job description.
    job_keywords : List[str]
        The list of top keywords extracted from the job description.
    resume_tokens : List[str]
        The list of preprocessed tokens extracted from the résumé.
    report_path : str
        Path to the output text file where the report will be written.
    """
    resume_token_set = set(resume_tokens)
    present = [kw for kw in job_keywords if kw in resume_token_set]
    missing = [kw for kw in job_keywords if kw not in resume_token_set]
    coverage = (len(present) / len(job_keywords)) * 100 if job_keywords else 0.0
    with open(report_path, 'w', encoding='utf-8') as fh:
        fh.write('MATCH REPORT\n')
        fh.write('============\n\n')
        fh.write(f'Cosine similarity score: {similarity:.4f}\n')
        fh.write(f'Keyword coverage: {coverage:.1f}% ({len(present)} of {len(job_keywords)} keywords)\n\n')
        fh.write('Job description keywords:\n')
        fh.write(', '.join(job_keywords) + '\n\n')
        fh.write('Present in résumé:\n')
        fh.write(', '.join(present) + '\n\n')
        fh.write('Missing from résumé:\n')
        fh.write(', '.join(missing) + '\n\n')
        fh.write('Notes:\n')
        fh.write('- The similarity score is a measure of textual overlap; higher is better.\n')
        fh.write('- The keyword coverage indicates how many of the top terms from the job description appear in your résumé.\n')
        fh.write('- Consider emphasising skills corresponding to missing keywords if they genuinely reflect your experience.\n')


def write_curated_resume(relevant_bullets: List[Tuple[str, int]],
                         other_bullets: List[str],
                         remaining_text: List[str],
                         curated_path: str) -> None:
    """Write the curated résumé to a file.

    The curated résumé consists of the relevant bullet points (sorted by
    keyword score), followed by the other bullet points and the remaining
    non‑bullet résumé text.  This function writes the result to
    ``curated_path``.

    Parameters
    ----------
    relevant_bullets : List[Tuple[str, int]]
        Bullet points paired with their keyword scores, sorted in descending
        order of relevance.
    other_bullets : List[str]
        Bullet points with no job keywords, kept in original order.
    remaining_text : List[str]
        Any lines from the résumé that were not classified as bullets, kept in
        original order.
    curated_path : str
        Path to the output text file where the curated résumé will be written.
    """
    with open(curated_path, 'w', encoding='utf-8') as fh:
        fh.write('CURATED RÉSUMÉ\n')
        fh.write('================\n\n')
        if relevant_bullets:
            fh.write('Key achievements aligned with the job description:\n')
            for bullet, score in relevant_bullets:
                fh.write(f'- {bullet} (matched {score} keyword' + ('s' if score > 1 else '') + ')\n')
            fh.write('\n')
        if other_bullets:
            fh.write('Other achievements:\n')
            for bullet in other_bullets:
                fh.write(f'- {bullet}\n')
            fh.write('\n')
        if remaining_text:
            fh.write('Additional information:\n')
            for line in remaining_text:
                fh.write(line.rstrip() + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Match a résumé against a job description and generate a match report.')
    parser.add_argument('resume', help='Path to the résumé file (PDF or .txt)')
    parser.add_argument('job_description', help='Path to the job description file (PDF or .txt)')
    parser.add_argument('--report', default='match_report.txt', help='Output path for the match report (default: match_report.txt)')
    parser.add_argument('--curated', default='curated_resume.txt', help='Output path for the curated résumé (default: curated_resume.txt)')
    parser.add_argument('--keywords', type=int, default=20, help='Number of top keywords to extract from the job description (default: 20)')
    args = parser.parse_args()

    # Extract text from both files
    resume_text = extract_text(args.resume)
    job_text = extract_text(args.job_description)

    # Preprocess both texts
    resume_tokens = preprocess(resume_text)
    job_tokens = preprocess(job_text)

    # Compute similarity using the cleaned tokens.  Joining tokens ensures that
    # the vectoriser sees the same preprocessed representation as used for
    # keyword extraction.
    similarity = compute_similarity(' '.join(resume_tokens), ' '.join(job_tokens))

    # Extract top keywords from job description
    job_keywords = get_top_keywords(job_tokens, args.keywords)

    # Sort bullets in résumé by relevance
    bullets = extract_bullets(resume_text)
    relevant_bullets, other_bullets = sort_bullets_by_relevance(bullets, job_keywords)
    # Capture remaining non‑bullet text (so it can be appended at the end).
    # We skip any lines that appear to be bullet points (based on the same
    # regex used in extract_bullets), ensuring that bullet content does not
    # appear twice in the curated output.
    remaining_lines: List[str] = []
    bullet_pattern = re.compile(r'^(\s*)(\d+\.\s*)?[\u2022\-\*]+\s+')
    for line in resume_text.splitlines():
        if not line.strip():
            continue
        if bullet_pattern.match(line):
            continue
        remaining_lines.append(line)

    # Write outputs
    write_report(similarity, job_keywords, resume_tokens, args.report)
    write_curated_resume(relevant_bullets, other_bullets, remaining_lines, args.curated)

    print(f'Report written to {args.report}')
    print(f'Curated résumé written to {args.curated}')


if __name__ == '__main__':
    main()