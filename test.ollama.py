import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "llama3.1:latest",
    "prompt": "Write a single LaTeX Beamer frame with one bullet point about linear regression.",
    "stream": False,
    "options": {
        "num_predict": 80,  # chota output, fast
    },
}

print("Calling Ollama...")
resp = requests.post(url, json=payload, timeout=90)
print("Status code:", resp.status_code)
data = resp.json()
print("Keys:", data.keys())
print("Response text snippet:\n", data.get("response", "")[:400])
import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "llama3.1:latest",
    "prompt": "Write a single LaTeX Beamer frame with one bullet point about linear regression.",
    "stream": False,
    "options": {
        "num_predict": 80,  # chota output, fast
    },
}

print("Calling Ollama...")
resp = requests.post(url, json=payload, timeout=90)
print("Status code:", resp.status_code)
data = resp.json()
print("Keys:", data.keys())
print("Response text snippet:\n", data.get("response", "")[:400])
