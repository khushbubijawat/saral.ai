# """
# Frames generator: high-level API to go from parsed sections to audience-specific frames
# using an open LLM via Ollama.

# BONUS ADDED:
# 1) Automated equation simplification (SymPy) for general audience.
# 2) Bilingual bullets (English + Hindi) for general audience.
# """

# from __future__ import annotations
# from typing import List, Dict, Any, Optional

# from .llm_client_ollama import call_ollama_structured, Audience


# # ------------------ BONUS HELPERS ------------------ #

# def _try_import_sympy():
#     try:
#         import sympy as sp  # type: ignore
#         return sp
#     except Exception:
#         return None


# def _try_import_parse_latex():
#     try:
#         from sympy.parsing.latex import parse_latex  # type: ignore
#         return parse_latex
#     except Exception:
#         return None


# def _simplify_equation_latex(eq_latex: str) -> str:
#     """
#     Attempt to simplify a LaTeX equation using SymPy.
#     If SymPy/parse_latex not available or fails, return original.

#     NOTE: sympy.parsing.latex.parse_latex may need antlr4.
#     This function is safe: it never crashes the pipeline.
#     """
#     if not eq_latex or not isinstance(eq_latex, str):
#         return eq_latex

#     sp = _try_import_sympy()
#     parse_latex = _try_import_parse_latex()
#     if sp is None or parse_latex is None:
#         return eq_latex

#     try:
#         expr = parse_latex(eq_latex)
#         simp = sp.simplify(expr)
#         return sp.latex(simp)
#     except Exception:
#         return eq_latex


# def _translate_to_hindi(text: str, model_name: str) -> str:
#     """
#     Translate text to Hindi using Ollama if available.
#     If ollama not installed / model fails, return original text.
#     """
#     if not text.strip():
#         return text

#     try:
#         import ollama  # type: ignore
#     except Exception:
#         return text

#     prompt = (
#         "Translate the following to simple Hindi (Devanagari). "
#         "Keep any math tokens, equation IDs, symbols as-is. "
#         "Return only the Hindi translation.\n\n"
#         f"TEXT:\n{text}"
#     )

#     try:
#         resp = ollama.chat(
#             model=model_name,
#             messages=[{"role": "user", "content": prompt}],
#             options={"temperature": 0.0},
#         )
#         out = resp["message"]["content"].strip()
#         return out if out else text
#     except Exception:
#         return text


# def _make_bilingual_bullets(bullets: List[str], model_name: str) -> List[str]:
#     """
#     For each English bullet, add Hindi in parentheses.
#     Example: "We use SGD." -> "We use SGD. (हम SGD का उपयोग करते हैं।)"
#     """
#     bilingual = []
#     for b in bullets:
#         hi = _translate_to_hindi(b, model_name=model_name)
#         if hi.strip() and hi.strip() != b.strip():
#             bilingual.append(f"{b} ({hi})")
#         else:
#             bilingual.append(b)
#     return bilingual


# def _build_equation_override_map(
#     section: Dict[str, Any],
#     equation_ids: List[str],
# ) -> Dict[str, str]:
#     """
#     Build {eq_id: simplified_latex} for requested equation IDs.
#     """
#     overrides: Dict[str, str] = {}
#     eq_list = section.get("equations", []) or []
#     eq_by_id = {}
#     for eq in eq_list:
#         if isinstance(eq, dict) and "id" in eq:
#             eq_by_id[eq["id"]] = eq.get("latex", "")

#     for eid in equation_ids:
#         raw = eq_by_id.get(eid)
#         if raw:
#             overrides[eid] = _simplify_equation_latex(raw)

#     return overrides


# # ------------------ MAIN API ------------------ #

# def generate_frames_for_audience(
#     sections: List[Dict[str, Any]],
#     audience: Audience,
#     model_name: str = "phi",
# ) -> List[Dict[str, Any]]:
#     """
#     High-level API used in the main pipeline.

#     Input:
#       - sections: parsed LaTeX sections, each like:
#           {
#             "id": "sec:intro",
#             "title": "Introduction",
#             "text": "...",
#             "equations": [ { "id": "eq:1", "latex": "..."} ]
#           }
#       - audience: "expert" | "graduate" | "general"
#       - model_name: e.g. "llama3.1" or "gemma2:9b"

#     Output:
#       - list of frame dicts:
#         {
#           "section_id": "...",
#           "frame_title": "...",
#           "bullets": [...],                     # bilingual for general
#           "equation_ids_to_show": [...],
#           "equation_overrides": {...}          # only for general
#         }
#     """
#     all_frames: List[Dict[str, Any]] = []

#     for section in sections:
#         section_id = section["id"]

#         structured = call_ollama_structured(
#             model_name=model_name,
#             section=section,
#             audience=audience,
#         )

#         frames = structured.get("frames", [])
#         for fr in frames:
#             bullets = fr.get("bullets", []) or []
#             eq_ids = fr.get("equation_ids_to_show", []) or []

#             equation_overrides: Optional[Dict[str, str]] = None

#             # BONUS-1 + BONUS-2 apply automatically for general audience
#             if audience == "general":
#                 bullets = _make_bilingual_bullets(bullets, model_name=model_name)
#                 equation_overrides = _build_equation_override_map(section, eq_ids)

#             out_frame = {
#                 "section_id": section_id,
#                 "frame_title": fr.get("frame_title", section.get("title", "")),
#                 "bullets": bullets,
#                 "equation_ids_to_show": eq_ids,
#             }

#             if equation_overrides:
#                 out_frame["equation_overrides"] = equation_overrides

#             all_frames.append(out_frame)

#     return all_frames




"""
Frames generator: high-level API to go from parsed sections to audience-specific frames
using an open LLM via Ollama.

This file keeps the old behavior but adds bonus support:
- bilingual=True   -> add Hindi bullets (English + Hindi)
"""

from __future__ import annotations
from typing import List, Dict, Any

from .llm_client_ollama import call_ollama_structured, Audience
import ollama  # pip install ollama

def _translate_bullets_to_hindi(model_name: str, bullets: List[str]) -> List[str]:
    """
    Translate bullets to Hindi using Ollama.
    Accepts either a JSON list or plain-text list from the model.
    On failure, returns empty list (caller will keep English only).
    """
    if not bullets:
        return []

    prompt = (
        "Translate the following slide bullets to simple Hindi (not too formal). "
        "Keep meaning the same. Return EITHER a JSON list of strings, or just one bullet per line.\n\n"
        f"Bullets:\n{bullets}\n\nOutput:"
    )
    try:
        resp = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        content = resp["message"]["content"].strip()

        # Try JSON first
        try:
            import json
            data = json.loads(content)
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            pass

        # Fallback: split plain text lines
        lines = [ln.strip("-• \t") for ln in content.splitlines() if ln.strip()]
        return lines if lines else []
    except Exception:
        return []


def generate_frames_for_audience(
    sections: List[Dict[str, Any]],
    audience: Audience,
    model_name: str = "phi:2.7b",
    bilingual: bool = False,
) -> List[Dict[str, Any]]:
    """
    High-level API used in the main pipeline.

    Input:
      - sections: parsed LaTeX sections
      - audience: "expert" | "graduate" | "general"
      - model_name: e.g. "llama3.1:latest" or "phi:2.7b"
      - bilingual: if True, add Hindi bullets.

    Output:
      - list of frame dicts.
    """
    all_frames: List[Dict[str, Any]] = []

    for section in sections:
        section_id = section["id"]

        structured = call_ollama_structured(
            model_name=model_name,
            section=section,
            audience=audience,
        )

        frames = structured.get("frames", [])
        for fr in frames:
            bullets = fr.get("bullets", []) or []
            eq_ids = fr.get("equation_ids_to_show", []) or []

            if bilingual:
                hi_bullets = _translate_bullets_to_hindi(model_name, bullets)
                if hi_bullets:
                    merged = []
                    for en, hi in zip(bullets, hi_bullets):
                        merged.append(en)
                        merged.append(f"(Hindi) {hi}")
                    if len(bullets) > len(hi_bullets):
                        merged.extend(bullets[len(hi_bullets):])
                    bullets = merged

            all_frames.append(
                {
                    "section_id": section_id,
                    "frame_title": fr.get("frame_title", section.get("title", "")),
                    "bullets": bullets,
                    "equation_ids_to_show": eq_ids,
                }
            )

    return all_frames
