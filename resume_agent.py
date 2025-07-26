#!/usr/bin/env python3
"""
resume_agent.py
=================

This module extends the basic résumé matcher in this repository by adding
contextual matching and richer output to help candidates tailor their CVs to
specific job descriptions.  It operates entirely on the local machine and does
not call out to any external services.

Overview
--------

The agent processes two inputs: a résumé (PDF or plain‐text) and a job
description.  It extracts text, tokenises and normalises it, and then
computes both keyword overlap and contextual similarity between every bullet
point in the résumé and the job description.  Contextual similarity is
estimated using TF–IDF vectorisation across all bullet points plus the job
description; this captures the relative importance of words and yields a
similarity score even when exact keywords do not overlap.  The script then
produces a match report and a curated résumé outline ranked by a weighted
combination of keyword and contextual relevance.

Although large language models or external NLP services could offer more
sophisticated rewriting, this implementation deliberately avoids any network
dependencies.  Instead, it surfaces missing keywords and suggests where they
might be incorporated.  Candidates can use these insights to adjust their
language and emphasise relevant experience.

Usage
-----

Run the script from the command line with two positional arguments: the path
to your résumé and the path to the job description.  Optional flags allow
customising the output locations and the number of top keywords to extract.

Example:

    python resume_agent.py my_resume.pdf job_description.txt --report my_report.txt --curated my_curated.txt

See the ``main`` function at the bottom of this file for further details.

Limitations
-----------

This tool uses relatively simple NLP techniques (tokenisation, TF–IDF and
cosine similarity) and does not infer deep semantics.  Use the suggestions
as a guide rather than definitive advice.  Always ensure that any proposed
additions accurately reflect your skills and experience.
"""

import argparse
import os
import re
import string
import subprocess
from collections import Counter
from typing import Iterable, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# A reusable stop word list.  We replicate the list from the original
# resume_matcher.py to avoid external downloads.  Feel free to extend this set
# as needed.
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
    command ``pdftotext`` to convert the PDF into plain UTF‑8 text.  Otherwise
    the file is assumed to be a UTF‑8 encoded text file and is read directly.
    Unsupported extensions will raise a ``ValueError``.

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
        try:
            output = subprocess.check_output(['pdftotext', '-q', path, '-'])
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
    """Normalise and tokenise text into a list of words.

    This function lowercases the text, removes punctuation, splits on
    whitespace and filters out stop words, single‑character tokens and purely
    numeric tokens.  The returned tokens are suitable for TF–IDF
    vectorisation and keyword extraction.

    Parameters
    ----------
    text : str
        Raw text to be processed.

    Returns
    -------
    List[str]
        A list of cleaned tokens.
    """
    # Lowercase
    text = text.lower()
    # Replace punctuation with spaces
    trans_table = str.maketrans({k: ' ' for k in string.punctuation})
    text = text.translate(trans_table)
    # Split into tokens
    tokens = text.split()
    # Filter out stop words, single‑character tokens and numeric tokens
    cleaned = [t for t in tokens if t not in STOP_WORDS and len(t) > 1 and not t.isdigit()]
    return cleaned


def extract_bullets(text: str) -> List[str]:
    """Extract bullet lines from a résumé.

    Bullet points are identified by lines starting with common bullet characters
    (``•``, ``-`` or ``*``) possibly preceded by whitespace or numbering.
    The bullet marker and leading/trailing whitespace are stripped from the
    returned strings.

    Parameters
    ----------
    text : str
        The raw résumé text.

    Returns
    -------
    List[str]
        A list of bullet point strings.
    """
    bullet_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if re.match(r'^(\d+\.\s*)?[\u2022\-\*]+\s+', stripped):
            cleaned = re.sub(r'^(\d+\.\s*)?[\u2022\-\*]+\s+', '', stripped).strip()
            if cleaned:
                bullet_lines.append(cleaned)
    return bullet_lines


def compute_document_similarity(resume_text: str, job_text: str) -> float:
    """Compute the cosine similarity between two documents.

    Both inputs should be preprocessed strings (i.e. tokens joined by spaces).

    Parameters
    ----------
    resume_text : str
        Text extracted from the résumé (preprocessed and joined).
    job_text : str
        Text extracted from the job description (preprocessed and joined).

    Returns
    -------
    float
        Cosine similarity between the two documents (0–1).
    """
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


def compute_bullet_similarity(bullets: List[str], job_description: str) -> List[float]:
    """Compute contextual similarity between each bullet and the job description.

    The similarity is estimated using TF–IDF vectors constructed over all bullet
    points and the job description.  Each bullet receives a score between 0 and 1.

    Parameters
    ----------
    bullets : List[str]
        The list of bullet strings.
    job_description : str
        The full job description text (preprocessed and joined).

    Returns
    -------
    List[float]
        A list of similarity scores for each bullet.
    """
    if not bullets:
        return []
    # Build corpus: all bullets + job description
    corpus = bullets + [job_description]
    vectoriser = TfidfVectorizer()
    tfidf_matrix = vectoriser.fit_transform(corpus)
    job_vec = tfidf_matrix[-1]
    bullet_vecs = tfidf_matrix[:-1]
    sims = cosine_similarity(bullet_vecs, job_vec)
    return sims.flatten().tolist()


def compute_bullet_keyword_scores(bullets: List[str], job_keywords: Iterable[str]) -> List[int]:
    """Count how many job keywords appear in each bullet.

    Parameters
    ----------
    bullets : List[str]
        The résumé bullet points.
    job_keywords : Iterable[str]
        Keywords extracted from the job description.

    Returns
    -------
    List[int]
        The number of keywords present in each bullet.
    """
    keyword_set = set(job_keywords)
    scores: List[int] = []
    for bullet in bullets:
        tokens = preprocess(bullet)
        score = sum(1 for token in tokens if token in keyword_set)
        scores.append(score)
    return scores


def rank_bullets(bullets: List[str], context_scores: List[float], keyword_scores: List[int]) -> List[Tuple[str, float, int, float]]:
    """Combine contextual and keyword relevance to rank bullets.

    This function normalises the two score types to the range [0, 1] and
    computes a weighted combination (70 % contextual, 30 % keyword) for each
    bullet.  It returns a list of tuples containing the bullet text, its
    combined score, its raw keyword score and its contextual similarity score.

    Parameters
    ----------
    bullets : List[str]
        The résumé bullet points.
    context_scores : List[float]
        Contextual similarity scores (0–1) for each bullet.
    keyword_scores : List[int]
        Keyword overlap counts for each bullet.

    Returns
    -------
    List[Tuple[str, float, int, float]]
        A list of (bullet, combined_score, keyword_score, context_score)
        sorted from highest to lowest combined_score.
    """
    if not bullets:
        return []
    max_context = max(context_scores) if context_scores else 1.0
    max_keyword = max(keyword_scores) if keyword_scores else 1
    ranked: List[Tuple[str, float, int, float]] = []
    for bullet, ctx, kw in zip(bullets, context_scores, keyword_scores):
        ctx_norm = ctx / max_context if max_context > 0 else 0.0
        kw_norm = kw / max_keyword if max_keyword > 0 else 0.0
        combined = 0.7 * ctx_norm + 0.3 * kw_norm
        ranked.append((bullet, combined, kw, ctx))
    # Sort by combined score descending
    ranked.sort(key=lambda x: -x[1])
    return ranked


def write_report(global_similarity: float,
                 job_keywords: List[str],
                 resume_tokens: List[str],
                 report_path: str,
                 bullet_rankings: List[Tuple[str, float, int, float]] = None) -> None:
    """Write a match report to a text file.

    The report summarises the overall similarity, keyword coverage and any
    missing keywords.  If bullet rankings are provided, the top five bullets
    by combined relevance are also listed with their individual scores.

    Parameters
    ----------
    global_similarity : float
        Cosine similarity between the entire résumé and job description.
    job_keywords : List[str]
        Keywords extracted from the job description.
    resume_tokens : List[str]
        Preprocessed résumé tokens.
    report_path : str
        File to write the report to.
    bullet_rankings : List[Tuple[str, float, int, float]], optional
        Ranked bullets with scores for inclusion in the report.
    """
    # Determine which job keywords appear in the résumé
    resume_token_set = set(resume_tokens)
    keywords_in_resume = [kw for kw in job_keywords if kw in resume_token_set]
    missing_keywords = [kw for kw in job_keywords if kw not in resume_token_set]

    with open(report_path, 'w', encoding='utf-8') as fh:
        fh.write('MATCH REPORT\n')
        fh.write('============\n\n')
        fh.write(f'Overall similarity score: {global_similarity:.3f}\n')
        fh.write(f'Job description keywords extracted ({len(job_keywords)} total):\n')
        fh.write(', '.join(job_keywords) + '\n\n')
        fh.write(f'Keywords present in your résumé ({len(keywords_in_resume)}):\n')
        fh.write(', '.join(keywords_in_resume) + '\n\n')
        fh.write(f'Keywords missing from your résumé ({len(missing_keywords)}):\n')
        fh.write(', '.join(missing_keywords) + '\n\n')
        if bullet_rankings:
            fh.write('Top relevant bullet points (combined contextual/keyword score):\n')
            for bullet, combined, kw_score, ctx_score in bullet_rankings[:5]:
                fh.write(f'- {bullet}\n')
                fh.write(f'    Combined score: {combined:.3f}, keyword matches: {kw_score}, contextual similarity: {ctx_score:.3f}\n')
            fh.write('\n')
        fh.write('Interpretation:\n')
        fh.write('  • A higher overall similarity suggests that your résumé language is generally aligned with the job description.\n')
        fh.write('  • Keywords marked as missing are important terms from the job description that do not appear in your résumé.\n')
        fh.write('    Consider incorporating them where they accurately reflect your experience.\n')
        fh.write('  • The top bullet list highlights achievements that are most aligned with the role.  You may wish to move these higher in your résumé.\n')


def write_curated_resume(ranked_bullets: List[Tuple[str, float, int, float]],
                         other_lines: List[str],
                         curated_path: str,
                         job_keywords: List[str]) -> None:
    """Write a curated résumé outline to a text file.

    The curated résumé contains bullets sorted by combined relevance, followed
    by any remaining non‐bullet lines in their original order.  For each
    bullet, we also append a list of missing keywords that could
    be incorporated.  This serves as a starting point for rewriting rather
    than an automatic rewrite.

    Parameters
    ----------
    ranked_bullets : List[Tuple[str, float, int, float]]
        Bullet points and their scores sorted by relevance.
    other_lines : List[str]
        Non‐bullet lines from the résumé to append after the ranked bullets.
    curated_path : str
        Destination file for the curated résumé.
    job_keywords : List[str]
        Keywords extracted from the job description for suggesting missing terms.
    """
    keyword_set = set(job_keywords)
    with open(curated_path, 'w', encoding='utf-8') as fh:
        fh.write('CURATED RÉSUMÉ OUTLINE\n')
        fh.write('======================\n\n')
        if ranked_bullets:
            fh.write('Key achievements ranked by relevance:\n')
            for bullet, combined, kw_score, ctx_score in ranked_bullets:
                fh.write(f'- {bullet}\n')
                # Suggest missing keywords for this bullet
                tokens = set(preprocess(bullet))
                missing = [kw for kw in job_keywords if kw not in tokens]
                if missing:
                    # Limit suggestion length to avoid overwhelming the user
                    suggestions = ', '.join(missing[:5])
                    fh.write(f'    (Consider incorporating: {suggestions})\n')
                fh.write(f'    Combined relevance score: {combined:.3f}\n')
            fh.write('\n')
        if other_lines:
            fh.write('Additional information (unchanged order):\n')
            for line in other_lines:
                fh.write(line.rstrip() + '\n')


def main() -> None:
    """Parse command line arguments and orchestrate the resume matching process."""
    parser = argparse.ArgumentParser(description='Advanced résumé matcher with contextual and keyword analysis.')
    parser.add_argument('resume', help='Path to the résumé file (PDF or .txt)')
    parser.add_argument('job_description', help='Path to the job description file (PDF or .txt)')
    parser.add_argument('--report', default='match_report.txt', help='Output path for the match report (default: match_report.txt)')
    parser.add_argument('--curated', default='curated_resume.txt', help='Output path for the curated résumé (default: curated_resume.txt)')
    parser.add_argument('--keywords', type=int, default=20, help='Number of top keywords to extract from the job description (default: 20)')
    args = parser.parse_args()

    # Extract raw text from files
    resume_text = extract_text(args.resume)
    job_text = extract_text(args.job_description)

    # Preprocess both texts
    resume_tokens = preprocess(resume_text)
    job_tokens = preprocess(job_text)

    # Compute overall document similarity
    global_similarity = compute_document_similarity(' '.join(resume_tokens), ' '.join(job_tokens))

    # Extract top keywords from job description
    job_keywords = get_top_keywords(job_tokens, args.keywords)

    # Extract bullets and non‑bullet lines from résumé
    bullets = extract_bullets(resume_text)
    # Non‑bullet lines (for later) – skip lines identified as bullet points
    remaining_lines: List[str] = []
    bullet_pattern = re.compile(r'^(\s*)(\d+\.\s*)?[\u2022\-\*]+\s+')
    for line in resume_text.splitlines():
        if not line.strip():
            continue
        if bullet_pattern.match(line):
            continue
        remaining_lines.append(line)

    # Compute bullet scores
    context_scores = compute_bullet_similarity([' '.join(preprocess(b)) for b in bullets], ' '.join(job_tokens))
    keyword_scores = compute_bullet_keyword_scores(bullets, job_keywords)
    ranked_bullets = rank_bullets(bullets, context_scores, keyword_scores)

    # Write outputs
    write_report(global_similarity, job_keywords, resume_tokens, args.report, ranked_bullets)
    write_curated_resume(ranked_bullets, remaining_lines, args.curated, job_keywords)

    print(f'Report written to {args.report}')
    print(f'Curated résumé outline written to {args.curated}')


if __name__ == '__main__':
    main()