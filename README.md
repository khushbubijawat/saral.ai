
# saral.ai
PPT ->  https://docs.google.com/presentation/d/1ivWTa8ivJ0M-PK5rUpctiLDu8qbUQ6-NwckjNe5yHU4/edit?usp=sharing
# Theme 1 — Beamer LaTeX Presentation Customizer (Math-aware)

This is a **minimal end-to-end pipeline** for SARAL Theme 1, using **open-source LLMs via Ollama**
(e.g. **Llama 3.1** or **Gemma 2**) to generate *audience-specific Beamer slides* from a LaTeX source.

## Features

- Simple LaTeX parser (sections + display equations).
- Links equations to sections.
- Calls a local open-weight LLM (Llama 3.1 or Gemma 2 via Ollama) to:
  - Generate **Expert**, **Graduate**, and **General** audience variants.
- Renders each variant to a standalone **Beamer `.tex` file** in `outputs/`.
- Basic evaluation script: readability + math density.

> ⚠️ This is a **reference skeleton**, not a production LaTeX parser.
> It works well on reasonably clean LaTeX papers/notes with `\section{}`, `\subsection{}` and display math (`\[ ... \]` or `$$ ... $$`).

---

## 1. Prerequisites

### 1.1. Install Ollama

Download & install from: https://ollama.com

Make sure `ollama` command works:

```bash
ollama --version
```

### 1.2. Pull an open model

You can use **either** Llama 3.1 or Gemma 2 (or both).

```bash
# Llama 3.1 (8B, good default)
ollama pull llama3.1

# OR Gemma 2 (e.g., 9B)
ollama pull gemma2:9b
```

### 1.3. Python dependencies

From project root:

```bash
pip install -r requirements.txt
```

---

## 2. Quick start

A small demo LaTeX file is in `data/sample_input.tex`.

Run the pipeline for a single audience, for example **expert + Llama 3.1**:

```bash
python -m src.main \
  --input data/sample_input.tex \
  --audience expert \
  --model llama3.1
```

This will produce something like:

- `outputs/slides_expert.tex`

Similarly for **graduate**:

```bash
python -m src.main --input data/sample_input.tex --audience graduate --model llama3.1
```

And for **general**:

```bash
python -m src.main --input data/sample_input.tex --audience general --model llama3.1
```

Or run all three variants at once:

```bash
python -m src.main --input data/sample_input.tex --audience all --model llama3.1
```

---

## 3. Compile Beamer outputs

From project root, for example:

```bash
cd outputs
pdflatex slides_expert.tex
pdflatex slides_graduate.tex
pdflatex slides_general.tex
```

(Or use `xelatex` if you prefer.)

---

## 4. Files overview

- `README.md` — this file.
- `requirements.txt` — Python dependencies.
- `src/`
  - `main.py` — CLI entry, orchestrates the pipeline.
  - `latex_parser.py` — simple LaTeX structure + equation extractor.
  - `llm_client_ollama.py` — wrapper around Ollama chat API.
  - `frames_generator.py` — `generate_frames_for_audience()` implementation.
  - `beamer_renderer.py` — convert frames → Beamer `.tex`.
  - `eval_metrics.py` — basic readability + math density metrics.
- `data/`
  - `sample_input.tex` — small example LaTeX source.
- `outputs/`
  - (Generated Beamer `.tex` files for each audience.)

---

## 5. Notes

- Parser is intentionally simple; you can replace `latex_parser.py` with a more powerful one
  using `latexwalker`, `plasTeX`, or `pandoc` if you want.
- The important part for the assignment is how we:
  1. Parse the LaTeX into a **structured representation**.
  2. Feed that representation into an **open-source LLM via Ollama**.
  3. Turn the LLM's structured output into **Beamer frames**.

Enjoy hacking ✨
>>>>>>> d584546 (Initial commit)
