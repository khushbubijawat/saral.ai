import ollama

def translate_to_hindi(text: str, model_name: str = "llama3.1") -> str:
    prompt = (
        "Translate the following into simple Hindi (keep technical terms in English):\n"
        f"{text}"
    )
    try:
        client = ollama.Client(timeout=60)
        resp = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        content = resp["message"]["content"].strip()
        return content or text
    except Exception:
        return text  # fallback
