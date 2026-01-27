# rag_engine/rag/answer_generator.py

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from rag_engine.rag.prompt_builder import build_prompt

load_dotenv()


class AnswerGenerator:
    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview",
        temperature: float = 0.0,
    ):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY in .env"
            )

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )

    def generate(self, question: str, contexts):
        prompt = build_prompt(question, contexts)
        response = self.llm.invoke(prompt)

        # 🔥 Normalize Gemini Flash output
        content = response.content

        if isinstance(content, list):
            # Extract text parts
            texts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    texts.append(part["text"])
                else:
                    texts.append(str(part))
            return "\n".join(texts)

        return str(content)
