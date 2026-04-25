import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openai import OpenAI
from config import LLM_API_URL, LLM_API_KEY, LLM_MODEL


def main():
    print("Testing LLM with config...\n")

    try:
        client = OpenAI(
            base_url=LLM_API_URL,
            api_key=LLM_API_KEY
        )

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is machine learning?"}
            ],
            temperature=0.2,
            max_tokens=200
        )

        answer = response.choices[0].message.content

        print("✅ LLM RESPONSE:\n")
        print(answer)

    except Exception as e:
        print("❌ ERROR:")
        print(e)


if __name__ == "__main__":
    main()