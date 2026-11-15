import os
import re
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

# Override via env var / Streamlit secret without touching code
MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")


def _clean(text: str) -> str:
    # Remove reasoning blocks some Qwen models emit
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def build_rag(retriever):

    llm = ChatGroq(
        model_name=MODEL,
        temperature=0.1,
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

        response = llm.invoke(prompt)

        return {
            "answer": _clean(response.content),
            "sources": [d.metadata.get("source_file") for d in docs]
        }

    return rag_answer
