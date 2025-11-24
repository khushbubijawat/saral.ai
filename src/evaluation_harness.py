# src/evaluation_harness.py

import re
import csv
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# -----------------------------
#  Common helpers
# -----------------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "at",
    "is", "are", "was", "were", "be", "this", "that", "with", "as",
    "by", "from", "it", "we", "you", "they", "he", "she", "its", "our",
    "their", "these", "those", "have", "has", "had", "do", "does", "did"
}


def normalize_token(token: str) -> str:
    token = token.lower()
    token = re.sub(r"[^a-z0-9]+", "", token)
    return token


def sentence_tokenize(text: str) -> List[str]:
    # very simple sentence splitter
    text = text.replace("\n", " ")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def tokenize_content(sentence: str) -> List[str]:
    tokens = re.findall(r"\w+", sentence.lower())
    tokens = [normalize_token(t) for t in tokens if normalize_token(t)]
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def jaccard_similarity(a_tokens: List[str], b_tokens: List[str]) -> float:
    set_a = set(a_tokens)
    set_b = set(b_tokens)
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


# -----------------------------
# 1. Content overlap metrics
# -----------------------------

@dataclass
class ContentOverlapResult:
    precision: float
    recall: float
    f1: float
    matched_original: int
    total_original: int
    matched_variant: int
    total_variant: int


def compute_content_overlap(
    original_text: str,
    variant_text: str,
    similarity_threshold: float = 0.5
) -> ContentOverlapResult:
    """
    Treat each sentence as a 'claim'.
    A variant sentence 'matches' an original sentence if Jaccard similarity
    over content tokens >= similarity_threshold.
    """
    original_sents = sentence_tokenize(original_text)
    variant_sents = sentence_tokenize(variant_text)

    original_tokens = [tokenize_content(s) for s in original_sents]
    variant_tokens = [tokenize_content(s) for s in variant_sents]

    total_original = len(original_tokens)
    total_variant = len(variant_tokens)

    # For each original sentence: is there ANY variant sentence that matches?
    matched_original_flags = [False] * total_original
    matched_variant_flags = [False] * total_variant

    for i, o_tok in enumerate(original_tokens):
        for j, v_tok in enumerate(variant_tokens):
            sim = jaccard_similarity(o_tok, v_tok)
            if sim >= similarity_threshold:
                matched_original_flags[i] = True
                matched_variant_flags[j] = True

    matched_original = sum(matched_original_flags)
    matched_variant = sum(matched_variant_flags)

    precision = matched_variant / total_variant if total_variant else 0.0
    recall = matched_original / total_original if total_original else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return ContentOverlapResult(
        precision=precision,
        recall=recall,
        f1=f1,
        matched_original=matched_original,
        total_original=total_original,
        matched_variant=matched_variant,
        total_variant=total_variant,
    )


# -----------------------------
# 2. Readability (Flesch–Kincaid) + Math density
# -----------------------------

@dataclass
class ReadabilityResult:
    num_sentences: int
    num_words: int
    num_syllables: int
    flesch_kincaid_grade: Optional[float]


VOWELS = "aeiouy"


def count_syllables_in_word(word: str) -> int:
    """
    Very rough heuristic syllable counter.
    Good enough for relative comparisons.
    """
    word = word.lower()
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 0

    syllables = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in VOWELS
        if is_vowel and not prev_vowel:
            syllables += 1
        prev_vowel = is_vowel

    # silent 'e' handling
    if word.endswith("e") and syllables > 1:
        syllables -= 1

    return max(syllables, 1)


def compute_readability(text: str) -> ReadabilityResult:
    sentences = sentence_tokenize(text)
    words = re.findall(r"\w+", text)

    num_sentences = max(len(sentences), 1)
    num_words = max(len(words), 1)

    num_syllables = sum(count_syllables_in_word(w) for w in words)

    # Flesch-Kincaid Grade Level
    fk_grade = (
        0.39 * (num_words / num_sentences)
        + 11.8 * (num_syllables / num_words)
        - 15.59
    )

    return ReadabilityResult(
        num_sentences=num_sentences,
        num_words=num_words,
        num_syllables=num_syllables,
        flesch_kincaid_grade=fk_grade,
    )


@dataclass
class MathDensityResult:
    equations: int
    words: int
    equations_per_100_words: float


MATH_PATTERNS = [
    r"\$(?:[^$]|\\\$)+\$",                   # $...$
    r"\$\$(?:[^$]|\\\$)+\$\$",              # $$...$$
    r"\\\[(.+?)\\\]",                        # \[ ... \]
    r"\\begin\{equation\}(.+?)\\end\{equation\}",
    r"\\begin\{align\}(.+?)\\end\{align\}",
    r"\\begin\{align\*\}(.+?)\\end\{align\*\}",
]


def compute_math_density(latex_text: str) -> MathDensityResult:
    """
    Count LaTeX-style math environments and normalize by 100 words of text.
    """
    eq_count = 0
    for pattern in MATH_PATTERNS:
        eq_count += len(re.findall(pattern, latex_text, flags=re.DOTALL))

    # word count of non-math part (rough)
    text_without_math = latex_text
    for pattern in MATH_PATTERNS:
        text_without_math = re.sub(pattern, " ", text_without_math, flags=re.DOTALL)

    words = re.findall(r"\w+", text_without_math)
    num_words = len(words)

    if num_words == 0:
        density = 0.0
    else:
        density = eq_count * 100.0 / num_words

    return MathDensityResult(
        equations=eq_count,
        words=num_words,
        equations_per_100_words=density,
    )


# -----------------------------
# 3. Manual human eval helpers
# -----------------------------

# CSV format expected:
# paper_id,audience_level,variant_id,rater_id,score,comments
#
# where:
#   paper_id: e.g., "ml_1", "math_1", "interdisc_1"
#   audience_level: "expert" / "intermediate" / "beginner"
#   variant_id: name of your variant, e.g. "expert_frames_v1"
#   rater_id: e.g. "r1","r2","r3"
#   score: integer 1-5
#   comments: free text


@dataclass
class ManualEvalStats:
    per_variant_average: Dict[str, float]
    per_audience_average: Dict[str, float]
    overall_average: float


def summarize_manual_eval(csv_path: str) -> ManualEvalStats:
    variant_scores: Dict[str, List[float]] = {}
    audience_scores: Dict[str, List[float]] = {}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                score = float(row["score"])
            except Exception:
                continue

            variant_id = row.get("variant_id", "").strip()
            audience_level = row.get("audience_level", "").strip()

            if variant_id:
                variant_scores.setdefault(variant_id, []).append(score)
            if audience_level:
                audience_scores.setdefault(audience_level, []).append(score)

    per_variant_avg = {
        k: (sum(v) / len(v)) for k, v in variant_scores.items() if v
    }
    per_audience_avg = {
        k: (sum(v) / len(v)) for k, v in audience_scores.items() if v
    }

    all_scores = [s for scores in variant_scores.values() for s in scores]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return ManualEvalStats(
        per_variant_average=per_variant_avg,
        per_audience_average=per_audience_avg,
        overall_average=overall,
    )





# ------------------------------------------------------------------
# 4.  FRAME-LEVEL CONTENT OVERLAP (REQUIRED BY eval_runner.py)
# ------------------------------------------------------------------

def compute_content_overlap_for_frames(
    original_sections: List[Dict],
    generated_frames: List[Dict],
    similarity_threshold: float = 0.5
):
    """
    Computes sentence-level content overlap between:
        (A) Original LaTeX text  (flattened from sections)
        (B) Generated slide-frame bullets (flattened)

    Uses Jaccard similarity (via compute_content_overlap).

    Returns: dict with precision, recall, f1, and claim counts.
    """

    # Flatten original text
    original_text = " ".join(
        sec.get("text", "") for sec in original_sections
    )

    # Flatten generated frame bullets
    generated_text = " ".join(
        " ".join(fr.get("bullets", [])) for fr in generated_frames
    )

    # Use existing function
    res = compute_content_overlap(
        original_text,
        generated_text,
        similarity_threshold=similarity_threshold
    )

    return {
        "precision": res.precision,
        "recall": res.recall,
        "f1": res.f1,
        "matched_original": res.matched_original,
        "total_original": res.total_original,
        "matched_variant": res.matched_variant,
        "total_variant": res.total_variant,
    }
