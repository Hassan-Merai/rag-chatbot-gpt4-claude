# backend/services/anthropic.py
import os
from typing import List
import anthropic
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

class ClaudeLLM:
    def __init__(self, model_name: str = "claude-opus-4-20250514"):

        self.model_name = model_name
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY in .env file.")

        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"[ClaudeLLM Error] {e}"