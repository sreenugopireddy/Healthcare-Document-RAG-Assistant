import streamlit as st

from src.vectorstore import load_index
from src.rag_chain import build_rag
from dotenv import load_dotenv
import os

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment")




# -----------------------------
# UI Header
# -----------------------------
st.set_page_config(page_title="Healthcare RAG Assistant", layout="wide")

st.title("🩺 Healthcare Document RAG Assistant")

st.info(
    "This assistant answers using only the provided medical documents. "
    "It is not a substitute for professional medical advice."
)


# -----------------------------
# Load Vector DB + RAG Pipeline
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_system():
    db = load_index()
    retriever = db.as_retriever(search_kwargs={"k": 3})
    rag = build_rag(retriever)
    return rag


rag = load_system()


# -----------------------------
# Query Input
# -----------------------------
q = st.text_input("Ask a healthcare question from your document set:")


# -----------------------------
# Run RAG
# -----------------------------
if q:

    with st.spinner("Searching medical knowledge base..."):
        result = rag(q)

    st.markdown("## ✅ Answer")
    st.write(result["answer"])

    st.markdown("## 📚 Sources Used")
    unique_sources = list(set(result["sources"]))

    for s in unique_sources:
        if s:
            st.write(f"- {s}")
