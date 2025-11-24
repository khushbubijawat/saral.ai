# """
# Theme 1 — Beamer LaTeX Presentation Customizer (math-aware)
# Open-source LLM version (Ollama + Llama 3.1 / Gemma 2).

# Usage examples (from project root):

#     python -m src.main --input data/sample_input.tex --audience expert --model llama3.1
#     python -m src.main --input data/sample_input.tex --audience graduate --model gemma2:9b
#     python -m src.main --input data/sample_input.tex --audience all --model llama3.1
# """

# from __future__ import annotations
# import argparse
# from pathlib import Path
# from typing import List, Dict

# from .latex_parser import parse_latex_file
# from .frames_generator import generate_frames_for_audience
# from .beamer_renderer import frames_to_beamer_tex
# from .eval_metrics import readability_score, math_density


# AUDIENCE_CHOICES = ["expert", "graduate", "general", "all"]


# def parse_args() -> argparse.Namespace:
#     ap = argparse.ArgumentParser(description="Theme 1 — Beamer LaTeX Presentation Customizer (Ollama-based).")
#     ap.add_argument(
#         "--input",
#         "-i",
#         required=True,
#         help="Path to input LaTeX file.",
#     )
#     ap.add_argument(
#         "--audience",
#         "-a",
#         choices=AUDIENCE_CHOICES,
#         default="expert",
#         help="Target audience (expert | graduate | general | all).",
#     )
#     ap.add_argument(
#         "--model",
#         "-m",
#         default="llama3.1",
#         help="Ollama model name (e.g. llama3.1, gemma2:9b, gemma2:27b).",
#     )
#     ap.add_argument(
#         "--output-dir",
#         "-o",
#         default="outputs",
#         help="Directory to write Beamer .tex files.",
#     )
#     ap.add_argument(
#         "--title",
#         "-t",
#         default=None,
#         help="Optional override for Beamer document title.",
#     )
#     return ap.parse_args()


# def run_for_audience(
#     input_path: Path,
#     audience: str,
#     model_name: str,
#     output_dir: Path,
#     title: str | None = None,
# ):
#     print(f"[*] Parsing LaTeX: {input_path}")
#     sections, equations_by_id = parse_latex_file(input_path)

#     print(f"[*] Parsed {len(sections)} sections, {len(equations_by_id)} equations.")

#     print(f"[*] Generating frames for audience='{audience}' using model='{model_name}' ...")
#     frames = generate_frames_for_audience(sections, audience=audience, model_name=model_name)

#     print(f"[*] Generated {len(frames)} frames. Evaluating ...")
#     fk_grade = readability_score(frames)
#     density = math_density(frames)
#     print(f"    - Flesch–Kincaid grade level: {fk_grade:.2f}")
#     print(f"    - Math density (eq per 100 words): {density:.2f}")

#     audience_suffix = audience
#     out_tex = output_dir / f"slides_{audience_suffix}.tex"
#     doc_title = title or f"{input_path.stem} ({audience.capitalize()} audience)"

#     print(f"[*] Rendering Beamer .tex → {out_tex}")
#     tex_str = frames_to_beamer_tex(frames, equations_by_id, title=doc_title)

#     out_tex.write_text(tex_str, encoding="utf-8")
#     print("[+] Done.")


# def main():
#     args = parse_args()

#     input_path = Path(args.input)
#     if not input_path.is_file():
#         raise SystemExit(f"Input LaTeX not found: {input_path}")

#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     audiences: List[str]
#     if args.audience == "all":
#         audiences = ["expert", "graduate", "general"]
#     else:
#         audiences = [args.audience]

#     for aud in audiences:
#         run_for_audience(
#             input_path=input_path,
#             audience=aud,
#             model_name=args.model,
#             output_dir=output_dir,
#             title=args.title,
#         )


# if __name__ == "__main__":
#     main()


"""
Theme 1 — Beamer LaTeX Presentation Customizer (math-aware)
Open-source LLM version (Ollama + Llama 3.1 / Gemma 2).

Usage examples (from project root):

    python -m src.main --input data/sample_input.tex --audience expert --model llama3.1
    python -m src.main --input data/sample_input.tex --audience graduate --model gemma2:9b
    python -m src.main --input data/sample_input.tex --audience all --model llama3.1

New (bonus):
    python -m src.main --input data/sample_input.tex --audience general --model phi:2.7b --bilingual
    python -m src.main --input data/sample_input.tex --audience general --model phi:2.7b --simplify-equations
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

from .latex_parser import parse_latex_file
from .frames_generator import generate_frames_for_audience
from .beamer_renderer import frames_to_beamer_tex
from .eval_metrics import readability_score, math_density

AUDIENCE_CHOICES = ["expert", "graduate", "general", "all"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Theme 1 — Beamer LaTeX Presentation Customizer (Ollama-based)."
    )
    ap.add_argument("--input", "-i", required=True, help="Path to input LaTeX file.")
    ap.add_argument(
        "--audience",
        "-a",
        choices=AUDIENCE_CHOICES,
        default="expert",
        help="Target audience (expert | graduate | general | all).",
    )
    ap.add_argument(
        "--model",
        "-m",
        default="llama3.1",
        help="Ollama model name (e.g. llama3.1, gemma2:9b, gemma2:27b, phi:2.7b).",
    )
    ap.add_argument(
        "--output-dir",
        "-o",
        default="outputs",
        help="Directory to write Beamer .tex files.",
    )
    ap.add_argument(
        "--title",
        "-t",
        default=None,
        help="Optional override for Beamer document title.",
    )
    ap.add_argument(
        "--preamble",
        default=None,
        help="Optional path to a LaTeX preamble/theme snippet to include.",
    )
    # -------- BONUS FLAGS (safe defaults) ----------
    ap.add_argument(
        "--bilingual",
        action="store_true",
        help="If set, also generate Hindi+English bilingual bullets/handout for general audience.",
    )
    ap.add_argument(
        "--simplify-equations",
        action="store_true",
        help="If set, simplify equations for lay/general variant using SymPy (when possible).",
    )
    return ap.parse_args()


def run_for_audience(
    input_path: Path,
    audience: str,
    model_name: str,
    output_dir: Path,
    title: str | None = None,
    bilingual: bool = False,
    simplify_equations: bool = False,
    extra_preamble: str | None = None,
):
    print(f"[*] Parsing LaTeX: {input_path}")
    sections, equations_by_id = parse_latex_file(
        input_path, simplify_equations=simplify_equations, audience=audience
    )

    print(f"[*] Parsed {len(sections)} sections, {len(equations_by_id)} equations.")

    print(f"[*] Generating frames for audience='{audience}' using model='{model_name}' ...")
    frames = generate_frames_for_audience(
        sections,
        audience=audience,
        model_name=model_name,
        bilingual=bilingual,
    )

    print(f"[*] Generated {len(frames)} frames. Evaluating ...")
    fk_grade = readability_score(frames)
    density = math_density(frames)
    print(f"    - Flesch–Kincaid grade level: {fk_grade:.2f}")
    print(f"    - Math density (eq per 100 words): {density:.2f}")

    audience_suffix = audience
    out_tex = output_dir / f"slides_{audience_suffix}.tex"
    doc_title = title or f"{input_path.stem} ({audience.capitalize()} audience)"

    print(f"[*] Rendering Beamer .tex → {out_tex}")
    tex_str = frames_to_beamer_tex(
        frames,
        equations_by_id,
        title=doc_title,
        extra_preamble=extra_preamble,
    )

    out_tex.write_text(tex_str, encoding="utf-8")
    print("[+] Done.")


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input LaTeX not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extra_preamble = (
        Path(args.preamble).read_text(encoding="utf-8") if args.preamble else None
    )

    if args.audience == "all":
        audiences: List[str] = ["expert", "graduate", "general"]
    else:
        audiences = [args.audience]

    for aud in audiences:
        run_for_audience(
            input_path=input_path,
            audience=aud,
            model_name=args.model,
            output_dir=output_dir,
            title=args.title,
            bilingual=args.bilingual,
            simplify_equations=args.simplify_equations,
            extra_preamble=extra_preamble,
        )


if __name__ == "__main__":
    main()
