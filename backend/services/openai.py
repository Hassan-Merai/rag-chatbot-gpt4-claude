# llms/gpt4_llm.py

import os
import openai
from typing import List

class GPT4LLM:
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate(self, prompt: str) -> str:
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response["choices"][0]["message"]["content"].strip()
