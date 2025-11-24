"""
Basic evaluation utilities:
- Flesch–Kincaid readability for text segments
- Math-density: equations per 100 words
"""

from __future__ import annotations
from typing import List, Dict

import re
import textstat


def collect_all_text(frames: List[Dict]) -> str:
    chunks = []
    for fr in frames:
        chunks.append(fr.get("frame_title", ""))
        for b in fr.get("bullets", []):
            chunks.append(b)
    return "\n".join(chunks)


def readability_score(frames: List[Dict]) -> float:
    """
    Compute Flesch–Kincaid grade level for all text from frames.
    """
    text = collect_all_text(frames)
    if not text.strip():
        return 0.0
    return textstat.flesch_kincaid_grade(text)


def math_density(frames: List[Dict]) -> float:
    """
    Equations per 100 words, approximating equations via [eq:*] placeholders
    in bullets/frame titles (if any appear) plus explicit equation lists.
    """
    text = collect_all_text(frames)
    words = re.findall(r"\w+", text)
    n_words = max(len(words), 1)

    # crude: count placeholders like [eq:1]
    n_eq_placeholders = len(re.findall(r"\[eq:\d+\]", text))

    return (n_eq_placeholders / n_words) * 100.0
