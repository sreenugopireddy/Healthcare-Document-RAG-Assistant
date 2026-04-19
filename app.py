import streamlit as st
import os
from dotenv import load_dotenv

from src.vectorstore import load_or_create_index
from src.rag_chain import build_rag

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Set it in .env file.")
    st.stop()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Healthcare RAG Assistant", layout="wide")

st.title("🩺 Healthcare Document RAG Assistant")

st.info(
    "This assistant answers using only the provided medical documents. "
    "It is not a substitute for professional medical advice."
)

# -----------------------------
# Load system (cached)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_system():
    db = load_or_create_index()
    retriever = db.as_retriever(search_kwargs={"k": 3})
    rag = build_rag(retriever)
    return rag

rag = load_system()

# -----------------------------
# Input
# -----------------------------
query = st.text_input("Ask a healthcare question:")

if query:
    with st.spinner("Searching..."):
        result = rag(query)

    st.subheader("✅ Answer")
    st.write(result["answer"])

    st.subheader("📚 Sources")
    for s in set(result["sources"]):
        if s:
            st.write(f"- {s}")