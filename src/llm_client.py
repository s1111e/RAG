import requests
from config import LLM_API_URL, LLM_API_KEY, LLM_MODEL, MAX_TOKENS, TEMPERATURE

def call_llm(messages):
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }

    try:
        response = requests.post(LLM_API_URL, headers=headers, json=data, timeout=60)
    except requests.RequestException as exc:
        print("LLM API request failed:", exc)
        return "Error"

    if response.status_code != 200:
        print("LLM API error:", response.text)
        return "Error"

    result = response.json()

    return result["choices"][0]["message"]["content"]