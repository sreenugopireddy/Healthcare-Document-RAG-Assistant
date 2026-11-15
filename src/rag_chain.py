import os
import re
import requests
from langchain_groq import ChatGroq

PROMPT_TEMPLATE = """
You are a healthcare assistant.

Answer ONLY using the provided context.
If answer is not found, say:
Not found in medical knowledge base.

Context:
{context}

Question:
{question}
"""

PREFERRED = ["qwen/qwen3.6-27b", "qwen/qwen3.8-27b", "qwen/qwen3-32b"]
SKIP_WORDS = ("whisper", "tts", "guard", "safeguard", "compound", "embed", "openai/")


def _available_models():
    key = os.getenv("GROQ_API_KEY")
    r = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {key}"},
        timeout=15,
    )
    r.raise_for_status()
    return [m["id"] for m in r.json()["data"]]


def pick_model():
    forced = os.getenv("GROQ_MODEL")
    ids = _available_models()
    if forced and forced in ids:
        return forced
    for m in PREFERRED:
        if m in ids:
            return m
    for m in ids:
        if not any(w in m.lower() for w in SKIP_WORDS):
            return m
    raise RuntimeError(f"No usable Groq model found. Available: {ids}")


MODEL = pick_model()


def _clean(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def build_rag(retriever):

    llm = ChatGroq(
        model_name=MODEL,
        temperature=0.1,
        max_tokens=400,            # stays under the 1000 output-tokens/min limit
        reasoning_effort="none",   # no thinking tokens, saves output budget
    )

    def rag_answer(question: str):
        docs = retriever.invoke(question)

        if not docs:
            return {
                "answer": "Not found in medical knowledge base.",
                "sources": []
            }

        context = "\n\n".join(d.page_content for d in docs)

        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )

        try:
            response = llm.invoke(prompt)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                return {
                    "answer": "Rate limit reached on the free Groq tier. Please wait about a minute and try again.",
                    "sources": []
                }
            raise

        return {
            "answer": _clean(response.content),
            "sources": [d.metadata.get("source_file") for d in docs]
        }

    return rag_answer
