# src/eval_runner.py

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

from .latex_parser import parse_latex_file
from .frames_generator import generate_frames_for_audience
from .eval_metrics import readability_score, math_density
from .evaluation_harness import (
    compute_content_overlap_for_frames,
    summarize_manual_eval,
    ManualEvalStats,
)

AUDIENCE_CHOICES = ["expert", "graduate", "general", "all"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluation harness for Beamer LaTeX Presentation Customizer."
    )
    ap.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input LaTeX file.",
    )
    ap.add_argument(
        "--audience",
        "-a",
        choices=AUDIENCE_CHOICES,
        default="all",
        help="Target audience (expert | graduate | general | all).",
    )
    ap.add_argument(
        "--model",
        "-m",
        default="llama3.1",
        help="Ollama model name (e.g. llama3.1, gemma2:9b, gemma2:27b).",
    )
    ap.add_argument(
        "--human-eval-csv",
        "-c",
        default=None,
        help="Optional path to manual human evaluation CSV to summarize.",
    )
    return ap.parse_args()


def run_eval_for_audience(
    input_path: Path,
    audience: str,
    model_name: str,
):
    print(f"\n=== Audience: {audience} ===")

    # 1) Parse LaTeX (same as main.py)
    print(f"[*] Parsing LaTeX: {input_path}")
    sections, equations_by_id = parse_latex_file(input_path)
    print(f"[*] Parsed {len(sections)} sections, {len(equations_by_id)} equations.")

    # 2) Generate frames (LLM call)
    print(f"[*] Generating frames for audience='{audience}' using model='{model_name}' ...")
    frames = generate_frames_for_audience(sections, audience=audience, model_name=model_name)
    print(f"[*] Generated {len(frames)} frames.")

    # 3) Automatic metrics (you already had these in main.py)
    fk_grade = readability_score(frames)
    density = math_density(frames)
    print(f"    - Flesch–Kincaid grade level: {fk_grade:.2f}")
    print(f"    - Math density (eq per 100 words): {density:.2f}")

    # 4) Content overlap vs original LaTeX
    overlap = compute_content_overlap_for_frames(
        original_sections=sections,
        generated_frames=frames,
        similarity_threshold=0.5,
    )
    print("    - Content overlap (sentence-level claims, Jaccard ≥ 0.5):")
    print(f"        · Precision: {overlap.precision:.3f} "
          f"({overlap.matched_variant}/{overlap.total_variant} variant sentences match)")
    print(f"        · Recall   : {overlap.recall:.3f} "
          f"({overlap.matched_original}/{overlap.total_original} original sentences covered)")
    print(f"        · F1       : {overlap.f1:.3f}")


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input LaTeX not found: {input_path}")

    # which audiences to run
    if args.audience == "all":
        audiences: List[str] = ["expert", "graduate", "general"]
    else:
        audiences = [args.audience]

    for aud in audiences:
        run_eval_for_audience(
            input_path=input_path,
            audience=aud,
            model_name=args.model,
        )

    # Optional: summarize manual human evaluation CSV
    if args.human_eval_csv:
        csv_path = Path(args.human_eval_csv)
        if not csv_path.is_file():
            print(f"[!] Human eval CSV not found: {csv_path}")
        else:
            print("\n=== Manual human evaluation summary ===")
            stats: ManualEvalStats = summarize_manual_eval(csv_path)
            print("Per-variant averages:")
            for v, s in stats.per_variant_average.items():
                print(f"    {v}: {s:.2f}")
            print("Per-audience averages:")
            for a, s in stats.per_audience_average.items():
                print(f"    {a}: {s:.2f}")
            print(f"Overall average score: {stats.overall_average:.2f}")


if __name__ == "__main__":
    main()
