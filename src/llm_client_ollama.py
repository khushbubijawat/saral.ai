# # """
# # LLM client for open-weight models served by Ollama (e.g., Llama 3.1, Gemma 2).

# # Requires:
# #     pip install ollama

# # Make sure Ollama is running and the model is pulled, e.g.:
# #     ollama pull llama3.1
# #     # or
# #     ollama pull gemma2:9b
# # """

# # from __future__ import annotations
# # import json
# # from typing import Literal, Dict, Any

# # import ollama

# # Audience = Literal["expert", "graduate", "general"]


# # SYSTEM_PROMPT = """\
# # You are a LaTeX Beamer slide writer for research talks.

# # You are given a parsed LaTeX section (title, prose text, equations with IDs).
# # Your job is to design slides for ONE audience type:
# # - expert: short high-level bullets, keep ALL math and advanced notation
# # - graduate: keep key equations, add more explanatory bullets and 1–2 line derivations
# # - general: minimal math (only simplest formulas), focus on intuition and analogies

# # Hard rules:
# # - ALWAYS respond in valid JSON ONLY. No extra text.
# # - JSON schema:

# # {
# #   "frames": [
# #     {
# #       "frame_title": "string",
# #       "bullets": ["bullet 1", "bullet 2", "..."],
# #       "equation_ids_to_show": ["eq:energy", "eq:loss"]  // may be empty
# #     },
# #     ...
# #   ]
# # }

# # - Do NOT invent new equations; only refer to equation IDs that appear in the input.
# # - Bullets must be short, LaTeX safe, and not exceed ~20 words each.
# # - For general audience, you may omit all equation IDs if math is too advanced.
# # """


# # def build_user_payload(section: Dict[str, Any], audience: Audience) -> str:
# #     """
# #     Convert a parsed Section (dict) into a compact JSON string for the LLM.
# #     """
# #     payload = {
# #         "audience": audience,
# #         "section": {
# #             "id": section["id"],
# #             "title": section["title"],
# #             "text": section["text"],
# #             "equations": section.get("equations", []),
# #         },
# #     }
# #     return json.dumps(payload, ensure_ascii=False, indent=2)


# # def call_ollama_structured(
# #     model: str,
# #     section: Dict[str, Any],
# #     audience: Audience,
# # ) -> Dict[str, Any]:
# #     """
# #     Call Ollama (llama3.1 / gemma2:9b / gemma2:27b / etc.) and expect strictly-JSON response.
# #     """
# #     user_content = build_user_payload(section, audience)

# #     response = ollama.chat(
# #         model=model,
# #         messages=[
# #             {"role": "system", "content": SYSTEM_PROMPT},
# #             {
# #                 "role": "user",
# #                 "content": (
# #                     "Here is the section data as JSON. "
# #                     "Generate Beamer frames for the given audience.\n\n"
# #                     f"{user_content}"
# #                 ),
# #             },
# #         ],
# #     )

# #     raw = response.message.content.strip()

# #     # Handle ```json ... ``` wrappers if model adds them
# #     if raw.startswith("```"):
# #         raw = raw.strip("`")
# #         if raw.lower().startswith("json"):
# #             raw = raw[4:].lstrip()

# #     try:
# #         data = json.loads(raw)
# #     except json.JSONDecodeError as e:
# #         raise ValueError(f"Model did not return valid JSON: {e}\n\nRAW:\n{raw}")

# #     return data






# ##### using local rule 
# from __future__ import annotations

# from typing import Literal

# AudienceType = Literal["expert", "graduate", "general"]


# def _summarise_text(text: str, max_sentences: int = 3) -> str:
#     """
#     Very naive text summariser:
#     - split into sentences
#     - take the first N
#     """
#     import re

#     sentences = re.split(r"(?<=[.!?])\s+", text.strip())
#     sentences = [s.strip() for s in sentences if s.strip()]
#     return " ".join(sentences[:max_sentences])


# def call_llm(prompt: str, audience: AudienceType) -> str:
#     """
#     Local, rule-based 'LLM' that generates different slide styles
#     for expert / graduate / general without any external API calls.
#     """
#     import re

#     # 1) Extract section title
#     m_title = re.search(r"Section title:\s*(.+)", prompt)
#     title = m_title.group(1).strip() if m_title else "Auto Slide"

#     # 2) Extract original text
#     m_text = re.search(
#         r"Original prose \(text paragraphs\):\s*(.+?)\n\nEquations",
#         prompt,
#         re.DOTALL,
#     )
#     orig_text = m_text.group(1).strip() if m_text else ""

#     # 3) Extract equations block
#     m_eq = re.search(
#         r"Equations .*?:\s*(.+?)\n\nKey structured content:",
#         prompt,
#         re.DOTALL,
#     )
#     eq_block = m_eq.group(1).strip() if m_eq else ""

#     equations = re.findall(
#         r"(\\begin\{equation\*?\}.*?\\end\{equation\*?\})",
#         eq_block,
#         re.DOTALL,
#     )

#     # 4) Simple summary based on audience
#     if audience == "expert":
#         summary = _summarise_text(orig_text, max_sentences=2)
#     elif audience == "graduate":
#         summary = _summarise_text(orig_text, max_sentences=3)
#     else:  # general
#         summary = _summarise_text(orig_text, max_sentences=2)

#     bullets = []

#     if audience == "expert":
#         bullets.append("High-level summary: " + summary)
#         for eq in equations:
#             bullets.append(f"Key equation: {eq}")
#         bullets.append("Assumes familiarity with basic concepts.")
#     elif audience == "graduate":
#         bullets.append("Main idea: " + summary)
#         if equations:
#             bullets.append("We focus on the core equation(s) and their role.")
#         bullets.append("Short explanation aimed at graduate-level readers.")
#     else:  # general
#         bullets.append("Intuitive idea: " + summary)
#         if equations:
#             bullets.append("We avoid the detailed formulas and focus on the concept.")
#         bullets.append("Intended for a general audience with minimal math background.")
#         bullets.append("% VISUAL: simple diagram illustrating the main concept")

#     # 5) Build Beamer frame
#     lines = [rf"\begin{{frame}}{{{title}}}", r"\begin{itemize}"]
#     for b in bullets:
#         lines.append("  \\item " + b)
#     lines.append(r"\end{itemize}")
#     lines.append(r"\end{frame}")

#     return "\n".join(lines)




################tiny 
# src/llm_client.py
# from __future__ import annotations

# import json
# from typing import Literal, Any
# from pathlib import Path
# from ctransformers import AutoModelForCausalLM

# Audience = Literal["expert", "graduate", "general"]

# # Path to the GGUF file (project root)
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# MODEL_PATH = PROJECT_ROOT / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# print(f"[llm_client] Loading TinyLlama from {MODEL_PATH} ...")

# llm = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH.parent,
#     model_file=MODEL_PATH.name,
#     model_type="llama",
#     gpu_layers=0,
# )

# print("[llm_client] Model loaded.")


# def _wrap_prompt(prompt: str, audience: Audience) -> str:
#     return (
#         f"You are generating *structured JSON* for a Beamer slide for audience '{audience}'.\n"
#         f"Return JSON ONLY.\n"
#         f"Format:\n"
#         f"{{\n"
#         f"  \"frames\": [\n"
#         f"    {{\n"
#         f"      \"frame_title\": \"...\",\n"
#         f"      \"bullets\": [\"...\", \"...\"],\n"
#         f"      \"equation_ids_to_show\": [1,2]\n"
#         f"    }}\n"
#         f"  ]\n"
#         f"}}\n\n"
#         f"{prompt}"
#     )


# def call_ollama_structured(model: str, section: Any, audience: Audience) -> dict:
#     """Return structured JSON required by frames_generator."""
    
#     # Build a prompt from section
#     if isinstance(section, dict):
#         title = section.get("title", "")
#         raw = section.get("raw", "") or section.get("text", "")
#         section_text = f"Section title: {title}\nContent: {raw}"
#     else:
#         section_text = str(section)

#     prompt = _wrap_prompt(section_text, audience)

#     # Generate output
#     output = llm(
#         prompt,
#         max_new_tokens=256,
#         temperature=0.3,
#         top_p=0.9,
#     )

#     output_str = str(output).strip()

#     # ---- TRY TO PARSE JSON ----
#     try:
#         data = json.loads(output_str)
#         if isinstance(data, dict) and "frames" in data:
#             return data
#     except Exception:
#         pass

#     # ---- SAFE FALLBACK (NO JSON) ----
#     print("[WARN] LLM returned non-JSON text. Using fallback wrapper.")

#     return {
#         "frames": [
#             {
#                 "frame_title": section.get("title", "Auto Slide") if isinstance(section, dict) else "Auto Slide",
#                 "bullets": [output_str],
#                 "equation_ids_to_show": [],
#             }
#         ]
#     }
#### tiny llm se



### phi se 
# src/llm_client.py
# src/llm_client_ollama.py

# from __future__ import annotations

# from typing import Literal, Dict, Any
# import json
# import re
# import textwrap

# import ollama  # make sure `pip install ollama` is done

# Audience = Literal["expert", "graduate", "general"]


# def _build_prompt(section: Dict[str, Any], audience: Audience) -> str:
#     """
#     Build a prompt for the LLM from one LaTeX section.
#     """
#     title = section.get("title", "")
#     text = section.get("text", "")
#     equations = section.get("equations", [])

#     eq_lines = []
#     for eq in equations:
#         eq_id = eq.get("id", "")
#         eq_tex = eq.get("latex", "")
#         if eq_id and eq_tex:
#             eq_lines.append(f"ID: {eq_id}\nLaTeX: {eq_tex}")

#     eq_block = "\n\n".join(eq_lines) if eq_lines else "None."

#     style_instructions = {
#         "expert": (
#             "- Keep almost all math.\n"
#             "- Use compact, high-level bullets.\n"
#             "- You can use advanced notation.\n"
#         ),
#         "graduate": (
#             "- Keep core equations.\n"
#             "- Add more explanatory bullets and short derivations.\n"
#         ),
#         "general": (
#             "- Minimize math (only very simple formulas).\n"
#             "- Focus on intuition and analogies.\n"
#         ),
#     }[audience]

#     prompt = f"""
# You are helping to design Beamer presentation slides from a LaTeX paper section.

# Section title: {title}

# Section text:
# \"\"\"{text}\"\"\"

# Equations (with IDs):
# {eq_block}

# Audience type: {audience}

# {style_instructions}

# Return a STRICT JSON object with this structure:

# {{
#   "frames": [
#     {{
#       "frame_title": "...",
#       "bullets": ["...", "..."],
#       "equation_ids_to_show": ["eq:1", "eq:2"]
#     }}
#   ]
# }}

# Rules:
# - Do NOT include any LaTeX markup in the JSON (no \\begin{{frame}}, no $...$).
# - 'equation_ids_to_show' must be chosen from the given IDs above.
# - 1–3 frames are enough for this section.
# - Do NOT wrap the JSON in markdown fences.
# """
#     return textwrap.dedent(prompt).strip()


# def _parse_json_from_response(content: str) -> Dict[str, Any]:
#     """
#     Try to extract a JSON object from the model output.
#     """
#     # 1) direct JSON
#     try:
#         data = json.loads(content)
#         if isinstance(data, dict):
#             return data
#     except Exception:
#         pass

#     # 2) try to grab the first {...} block
#     start = content.find("{")
#     end = content.rfind("}")
#     if start != -1 and end != -1 and end > start:
#         snippet = content[start : end + 1]
#         try:
#             data = json.loads(snippet)
#             if isinstance(data, dict):
#                 return data
#         except Exception:
#             pass

#     return {}


# def call_ollama_structured(
#     *,
#     model_name: str,
#     section: Dict[str, Any],
#     audience: Audience,
# ) -> Dict[str, Any]:
#     """
#     EXACTLY matches how frames_generator.py is calling it:
#       call_ollama_structured(model_name=model_name, section=section, audience=audience)

#     Returns:
#       {
#         "frames": [
#           {
#             "frame_title": str,
#             "bullets": [str, ...],
#             "equation_ids_to_show": [str, ...]
#           },
#           ...
#         ]
#       }
#     """
#     prompt = _build_prompt(section, audience)

#     try:
#         resp = ollama.chat(
#             model=model_name,
#             messages=[{"role": "user", "content": prompt}],
#             options={"temperature": 0.3, "top_p": 0.9},
#         )
#         content = resp["message"]["content"]
#         data = _parse_json_from_response(content)
#         if "frames" in data and isinstance(data["frames"], list):
#             return data
#         else:
#             print("[llm_client_ollama] Warning: model response not valid JSON structure, falling back.")
#     except Exception as e:
#         print(f"[llm_client_ollama] Error contacting Ollama ({e}), using rule-based fallback.")

#     # --------- Fallback: rule-based 1-frame summary (no LLM needed) ----------
#     title = section.get("title", "Overview")
#     text = section.get("text", "")
#     equations = section.get("equations", [])

#     # basic bullets: split by sentence
#     sentences = [s.strip() for s in re.split(r"[.\n]+", text) if s.strip()]
#     bullets = sentences[:5] if sentences else ([text] if text else [])

#     eq_ids = [eq.get("id") for eq in equations if isinstance(eq, dict) and "id" in eq]

#     return {
#         "frames": [
#             {
#                 "frame_title": title,
#                 "bullets": bullets,
#                 "equation_ids_to_show": eq_ids,
#             }
#         ]
#     }






################new version
from __future__ import annotations

from typing import Literal, Dict, Any
import json
import re
import textwrap
import ollama

Audience = Literal["expert", "graduate", "general"]
DEFAULT_TIMEOUT_SEC = 180   # 3 min timeout per section


def _build_prompt(section: Dict[str, Any], audience: Audience) -> str:
    """
    Build a prompt for the LLM from one LaTeX section.
    Includes simplified equations if present.
    """
    title = section.get("title", "")
    text = section.get("text", "")
    equations = section.get("equations", [])

    eq_lines = []
    for eq in equations:
        eq_id = eq.get("id", "")
        eq_tex = eq.get("latex_for_prompt") or eq.get("latex", "")
        if eq_id and eq_tex:
            eq_lines.append(f"ID: {eq_id}\nLaTeX: {eq_tex}")

    eq_block = "\n\n".join(eq_lines) if eq_lines else "None."

    style_instructions = {
        "expert": (
            "- Keep almost all math.\n"
            "- Use compact, high-level bullets.\n"
            "- You can use advanced notation.\n"
        ),
        "graduate": (
            "- Keep core equations.\n"
            "- Provide intuitive explanations + steps.\n"
        ),
        "general": (
            "- Use minimum math.\n"
            "- Focus on intuition, analogies, visuals.\n"
        ),
    }[audience]

    prompt = f"""
You are helping create Beamer presentation slides from a LaTeX section.

Section title: {title}

Raw text:
\"\"\"{text}\"\"\"

Equations:
{eq_block}

Audience type: {audience}

{style_instructions}

Return STRICT JSON ONLY:

{{
  "frames": [
    {{
      "frame_title": "...",
      "bullets": ["...", "..."],
      "equation_ids_to_show": ["eq:1"]
    }}
  ]
}}

Rules:
- DO NOT use LaTeX markup in JSON.
- equation_ids_to_show must be valid IDs from above.
- Do NOT wrap JSON in ``` or any other markdown fence.
"""
    return textwrap.dedent(prompt).strip()


def _parse_json_from_response(content: str) -> Dict[str, Any]:
    """
    Extract a JSON object even if model wraps it.
    """
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = content[start:end+1]
        try:
            data = json.loads(snippet)
            if isinstance(data, dict):
                return data
        except:
            pass

    return {}


def call_ollama_structured(
    *,
    model_name: str,
    section: Dict[str, Any],
    audience: Audience,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
) -> Dict[str, Any]:
    """
    Safe Ollama wrapper with timeout + fallback.
    """
    prompt = _build_prompt(section, audience)

    try:
        client = ollama.Client(host="http://localhost:11434", timeout=timeout_sec)
        resp = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "top_p": 0.9},
        )
        content = resp["message"]["content"]

        data = _parse_json_from_response(content)
        if "frames" in data and isinstance(data["frames"], list):
            return data

        print("[llm] Warning: model responded but JSON invalid. Falling back.")

    except Exception as e:
        print(f"[llm] Error contacting Ollama: {e}. Using fallback.")

    # ---------------- FALLBACK ------------------
    title = section.get("title", "Overview")
    text = section.get("text", "").strip()
    bullets = [s.strip() for s in re.split(r"[.\n]+", text) if s.strip()][:5]
    equations = section.get("equations", [])
    eq_ids = [eq.get("id") for eq in equations]

    return {
        "frames": [
            {
                "frame_title": title,
                "bullets": bullets or [text],
                "equation_ids_to_show": eq_ids,
            }
        ]
    }
