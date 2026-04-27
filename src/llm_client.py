import os
from config import LLM_API_URL, LLM_API_KEY, LLM_MODEL, MAX_TOKENS, TEMPERATURE
from openai import OpenAI

def call_llm(messages):
    try:
        client = OpenAI(
            base_url=LLM_API_URL,
            api_key=LLM_API_KEY
        )
        
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        return response.choices[0].message.content
    
    except Exception as exc:
        print("LLM API error:", str(exc))
        return "Error"