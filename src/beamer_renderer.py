"""
Convert frames into a full Beamer .tex document (safe & fixed).

UPDATED for BONUS-1:
- Supports per-frame equation overrides (simplified latex) via "equation_overrides".
"""

from __future__ import annotations
from typing import List, Dict
import re


def _latex_escape(s: str) -> str:
    """Escape LaTeX special characters in text fields like title/bullets."""
    if not s:
        return s
    s = s.replace('\\', r'\textbackslash{}')
    s = s.replace('_', r'\_')
    s = s.replace('%', r'\%')
    s = s.replace('&', r'\&')
    s = s.replace('#', r'\#')
    s = s.replace('{', r'\{')
    s = s.replace('}', r'\}')
    s = s.replace('$', r'\$')
    s = s.replace('^', r'\textasciicircum{}')
    s = s.replace('~', r'\textasciitilde{}')
    return s


def _strip_math_delimiters(eq: str) -> str:
    """
    Remove outer math delimiters if present: \[...\], $$...$$, $...$
    Keep inner content only.
    """
    eq = eq.strip()

    if eq.startswith(r"\[") and eq.endswith(r"\]"):
        eq = eq[2:-2].strip()

    if eq.startswith("$$") and eq.endswith("$$"):
        eq = eq[2:-2].strip()

    if eq.startswith("$") and eq.endswith("$"):
        eq = eq[1:-1].strip()

    return eq


def frames_to_beamer_tex(
    frames: List[Dict],
    equations_by_id: Dict[str, str],
    title: str | None = None,
    extra_preamble: str | None = None,
) -> str:
    """
    Convert list of frames + equations dict into a Beamer .tex string.
    """

    safe_title = _latex_escape(title) if title else "Auto-generated Slides"

    parts = [rf"""
\documentclass{{beamer}}

% Basic packages
\usepackage{{amsmath, amssymb}}
\usepackage[utf8]{{inputenc}}
\usetheme{{Madrid}}

\title{{{safe_title}}}
\author{{Theme 1 Demo}}
\date{{\today}}
"""]

    if extra_preamble:
        parts.append(extra_preamble)

    parts.append(r"\begin{document}")
    parts.append(r"\frame{\titlepage}")

    for fr in frames:
        frame_title = _latex_escape(fr.get("frame_title", ""))
        parts.append(rf"\begin{{frame}}{{{frame_title}}}")

        bullets = fr.get("bullets", [])
        if bullets:
            parts.append(r"\begin{itemize}")
            for b in bullets:
                parts.append(r"\item " + _latex_escape(b))
            parts.append(r"\end{itemize}")

        # NEW: equation_overrides support
        overrides = fr.get("equation_overrides", {}) or {}

        for eid in fr.get("equation_ids_to_show", []):
            eq_latex = overrides.get(eid) or equations_by_id.get(eid)
            if not eq_latex:
                continue

            eq_clean = _strip_math_delimiters(eq_latex)

            # If equation already has an environment, don't wrap
            if re.search(r"\\begin\{(equation|align|gather|multline)\*?\}", eq_latex):
                parts.append(eq_latex)
            else:
                parts.append(r"\[ " + eq_clean + r" \]")

        parts.append(r"\end{frame}")

    parts.append(r"\end{document}")

    return "\n".join(parts)
