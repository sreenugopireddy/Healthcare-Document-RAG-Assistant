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

def build_rag(retriever):

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    def rag_answer(question: str):

        docs = retriever.invoke(question)

        context = "\n\n".join(d.page_content for d in docs)

        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )

        response = llm.invoke(prompt)

        return {
            "answer": response.content,
            "sources": [d.metadata.get("source_file") for d in docs]
        }

    return rag_answer
