# """
# Simple LaTeX parser for Theme 1 demo.

# This is NOT a full LaTeX parser. It:
# - detects \section / \subsection titles
# - collects plain text per section
# - extracts display equations delimited by:
#   - \[ ... \]
#   - $$ ... $$

# You can later swap this with latexwalker / plasTeX / pandoc as needed.
# """

# from __future__ import annotations
# from dataclasses import dataclass, asdict
# from typing import List, Dict, Tuple
# import re
# from pathlib import Path


# @dataclass
# class Equation:
#     id: str
#     latex: str


# @dataclass
# class Section:
#     id: str
#     title: str
#     text: str
#     equations: List[Equation]


# SECTION_RE = re.compile(r"^\\section\{(?P<title>.+?)\}")
# SUBSECTION_RE = re.compile(r"^\\subsection\{(?P<title>.+?)\}")

# # Display math patterns
# DISPLAY_MATH_PATTERNS = [
#     (re.compile(r"\\\[(.+?)\\\]", re.DOTALL), "\[", "\]"),
#     (re.compile(r"\$\$(.+?)\$\$", re.DOTALL), "$$", "$$"),
# ]


# def _extract_display_equations(text: str, start_index: int = 0) -> Tuple[List[Equation], str]:
#     """
#     Extract display equations from a section text and assign IDs.
#     Returns (equations, cleaned_text).
#     """
#     equations: List[Equation] = []

#     # We'll replace equations with placeholders in cleaned_text, but we keep LaTeX content.
#     cleaned_text = text
#     eq_counter = start_index

#     for pattern, left, right in DISPLAY_MATH_PATTERNS:
#         # We will re-scan after each replacement in a loop
#         while True:
#             m = pattern.search(cleaned_text)
#             if not m:
#                 break
#             eq_counter += 1
#             body = m.group(1).strip()
#             eq_id = f"eq:{eq_counter}"
#             eq_latex = f"{left} {body} {right}"
#             equations.append(Equation(id=eq_id, latex=eq_latex))

#             # For simplicity, keep a short placeholder in prose
#             placeholder = f"[{eq_id}]"
#             cleaned_text = cleaned_text[: m.start()] + placeholder + cleaned_text[m.end() :]

#     return equations, cleaned_text


# def parse_latex_file(path: str | Path) -> Tuple[List[Dict, Dict[str, str]]]:
#     """
#     Parse a LaTeX file into a list of sections and a dict of equation_id -> latex.

#     Returns:
#         sections_as_dicts, equations_by_id
#     """
#     path = Path(path)
#     text = path.read_text(encoding="utf-8")

#     lines = text.splitlines()

#     sections: List[Section] = []
#     current_title = "Introduction"
#     current_id = "sec:intro"
#     current_buf: List[str] = []
#     eq_start_index = 0

#     def flush_section():
#         nonlocal eq_start_index
#         if not current_buf:
#             return
#         raw = "\n".join(current_buf).strip()
#         if not raw:
#             return
#         eqs, cleaned = _extract_display_equations(raw, start_index=eq_start_index)
#         if eqs:
#             eq_start_index = int(eqs[-1].id.split(":")[1])
#         sections.append(
#             Section(
#                 id=current_id,
#                 title=current_title,
#                 text=cleaned,
#                 equations=eqs,
#             )
#         )
#         current_buf.clear()

#     for line in lines:
#         line_stripped = line.strip()

#         m_sec = SECTION_RE.match(line_stripped)
#         m_subsec = SUBSECTION_RE.match(line_stripped)

#         if m_sec:
#             # flush previous section
#             flush_section()
#             current_title = m_sec.group("title")
#             current_id = "sec:" + re.sub(r"\W+", "_", current_title.lower()).strip("_")
#             continue

#         if m_subsec:
#             flush_section()
#             current_title = m_subsec.group("title")
#             current_id = "subsec:" + re.sub(r"\W+", "_", current_title.lower()).strip("_")
#             continue

#         current_buf.append(line)

#     # Flush last section
#     flush_section()

#     # Build equations_by_id
#     equations_by_id: Dict[str, str] = {}
#     for sec in sections:
#         for eq in sec.equations:
#             equations_by_id[eq.id] = eq.latex

#     sections_dicts = [asdict(sec) for sec in sections]
#     return sections_dicts, equations_by_id


# if __name__ == "__main__":
#     # Simple smoke test
#     import json
#     import sys

#     if len(sys.argv) < 2:
#         print("Usage: python -m src.latex_parser path/to/input.tex")
#         raise SystemExit(1)

#     secs, eqs = parse_latex_file(sys.argv[1])
#     print("Sections:")
#     print(json.dumps(secs, indent=2, ensure_ascii=False))
#     print("\nEquations:")
#     print(json.dumps(eqs, indent=2, ensure_ascii=False))



"""
Simple LaTeX parser for Theme 1 demo (math-aware, bonus-ready).

Goals:
- Detect sections/subsections and collect their text.
- Extract equations (display + common environments) and map IDs -> LaTeX.
- Keep the old API:
      sections, equations_by_id = parse_latex_file(path)
- Add OPTIONAL bonus knobs:
      parse_latex_file(path, simplify_equations=True, audience="general")

Notes:
- This is still a lightweight parser (regex-based), not a full TeX AST.
- It is designed to be robust on WPS/Word-exported LaTeX too.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Union
import re


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Equation:
    id: str
    latex: str

@dataclass
class Section:
    id: str
    title: str
    text: str
    equations: List[Equation]


# -----------------------------
# Regex patterns
# -----------------------------

SECTION_RE = re.compile(r"\\(sub)*section\{(.+?)\}")
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")

# Display math patterns we consider "equations"
DISPLAY_MATH_PATTERNS = [
    re.compile(r"\\\[(.+?)\\\]", re.DOTALL),                # \[ ... \]
    re.compile(r"\$\$(.+?)\$\$", re.DOTALL),               # $$ ... $$
    re.compile(r"\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}", re.DOTALL),
    re.compile(r"\\begin\{align\*?\}(.+?)\\end\{align\*?\}", re.DOTALL),
    re.compile(r"\\begin\{gather\*?\}(.+?)\\end\{gather\*?\}", re.DOTALL),
    re.compile(r"\\begin\{multline\*?\}(.+?)\\end\{multline\*?\}", re.DOTALL),
]

COMMENT_RE = re.compile(r"(?<!\\)%.*")   # strip TeX comments (not escaped)


def _strip_comments(tex: str) -> str:
    return re.sub(COMMENT_RE, "", tex)

def _clean_text(tex: str) -> str:
    """Remove most TeX commands for plain-text prompts."""
    tex = re.sub(r"\\(begin|end)\{.*?\}", " ", tex)
    tex = re.sub(r"\\\[.*?\\\]", " ", tex, flags=re.DOTALL)
    tex = re.sub(r"\$\$.*?\$\$", " ", tex, flags=re.DOTALL)
    tex = re.sub(r"\$.*?\$", " ", tex, flags=re.DOTALL)
    tex = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", tex)
    tex = re.sub(r"\s+", " ", tex)
    return tex.strip()

def _strip_outer_math(eq: str) -> str:
    eq = eq.strip()
    if eq.startswith(r"\[") and eq.endswith(r"\]"):
        eq = eq[2:-2].strip()
    if eq.startswith("$$") and eq.endswith("$$"):
        eq = eq[2:-2].strip()
    if eq.startswith("$") and eq.endswith("$"):
        eq = eq[1:-1].strip()
    return eq

def _simplify_latex_equation(eq_latex: str) -> str:
    """
    Try SymPy simplification, if SymPy + latex2sympy2 are available.
    If not, return original equation unchanged.
    """
    try:
        import sympy as sp  # pip install sympy
        try:
            from latex2sympy2 import latex2sympy  # pip install latex2sympy2
        except Exception:
            latex2sympy = None

        inner = _strip_outer_math(eq_latex)

        if latex2sympy is None:
            return eq_latex  # no converter installed

        expr = latex2sympy(inner)
        simp = sp.simplify(expr)
        return sp.latex(simp)
    except Exception:
        return eq_latex


# -----------------------------
# Public API
# -----------------------------

def parse_latex_file(
    input_path: Union[str, Path],
    simplify_equations: bool = False,
    audience: str = "expert",
) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Parse LaTeX into:
      sections: list of dicts with id/title/text/equations
      equations_by_id: dict mapping eq_id -> latex

    Old calls still work:
      parse_latex_file("x.tex")

    New optional args:
      parse_latex_file("x.tex", simplify_equations=True, audience="general")
    """
    path = Path(input_path)
    tex = path.read_text(encoding="utf-8", errors="ignore")
    tex = _strip_comments(tex)

    matches = list(SECTION_RE.finditer(tex))
    if not matches:
        matches = [re.match(r"", tex)]  # pseudo one section

    sections: List[Section] = []
    equations_by_id: Dict[str, str] = {}
    eq_counter = 1

    def next_eq_id() -> str:
        nonlocal eq_counter
        eid = f"eq:{eq_counter}"
        eq_counter += 1
        return eid

    boundaries = []
    for m in matches:
        if m is None:
            continue
        boundaries.append((m.start(), m.end(), m.group(2)))
    boundaries.append((len(tex), len(tex), ""))  # sentinel

    for idx in range(len(boundaries) - 1):
        start, end, title = boundaries[idx]
        next_start, _, _ = boundaries[idx + 1]
        body = tex[end:next_start]

        sec_id = f"sec:{idx+1}"
        eqs: List[Equation] = []

        # Extract equations
        for pat in DISPLAY_MATH_PATTERNS:
            for em in pat.finditer(body):
                inner_eq = em.group(1).strip()

                label_m = LABEL_RE.search(inner_eq)
                if label_m:
                    eid = label_m.group(1).strip()
                    inner_eq = LABEL_RE.sub("", inner_eq).strip()
                else:
                    eid = next_eq_id()

                eq_latex = inner_eq
                if simplify_equations and audience == "general":
                    eq_latex = _simplify_latex_equation(eq_latex)

                eqs.append(Equation(id=eid, latex=eq_latex))
                equations_by_id[eid] = eq_latex

        # Remove equations blocks from text
        body_no_math = body
        for pat in DISPLAY_MATH_PATTERNS:
            body_no_math = pat.sub(" ", body_no_math)

        text_clean = _clean_text(body_no_math)

        sections.append(
            Section(
                id=sec_id,
                title=title.strip() if title else f"Section {idx+1}",
                text=text_clean,
                equations=eqs,
            )
        )

    return [asdict(sec) for sec in sections], equations_by_id
