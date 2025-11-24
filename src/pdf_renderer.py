# pdf_renderer.py
"""
Utility to compile .tex → .pdf using pdflatex or xelatex.
Cross-platform support (Windows MiKTeX + Linux + Mac).
"""

import os
import subprocess
from pathlib import Path


def compile_pdf(tex_path: str, output_dir: str = None) -> str:
    tex_path = Path(tex_path).resolve()

    if output_dir is None:
        output_dir = tex_path.parent
    output_dir = Path(output_dir).resolve()

    pdf_path = output_dir / (tex_path.stem + ".pdf")

    # latex command selection
    LATEX_ENGINES = ["pdflatex", "xelatex"]

    for engine in LATEX_ENGINES:
        try:
            print(f"[*] Trying engine: {engine}")

            cmd = [
                engine,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={output_dir}",
                str(tex_path),
            ]

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if proc.returncode == 0 and pdf_path.exists():
                print(f"[+] PDF generated: {pdf_path}")
                return str(pdf_path)
            else:
                print(f"[!] {engine} failed. Checking next engine...")

        except FileNotFoundError:
            print(f"[!] {engine} not found on system.")

    raise RuntimeError(
        "No LaTeX engine (pdflatex/xelatex) succeeded.\n"
        "Install MiKTeX or TeXLive."
    )
