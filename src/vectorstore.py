import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.loader import load_documents
from src.chunker import chunk_docs

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "medical_index"

_embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"}
)


def build_index():
    print("⚡ Building new FAISS index...")

    docs = load_documents()
    chunks = chunk_docs(docs)

    db = FAISS.from_documents(chunks, _embeddings)
    db.save_local(INDEX_PATH)

    print("✅ Index built successfully")
    return db


def load_or_create_index():

    index_file = os.path.join(INDEX_PATH, "index.faiss")

    if not os.path.exists(index_file):
        return build_index()

    print("✅ Loading existing index...")
    return FAISS.load_local(
        INDEX_PATH,
        _embeddings,
        allow_dangerous_deserialization=True
    )