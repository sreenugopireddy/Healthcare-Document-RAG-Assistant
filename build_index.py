from src.loader import load_documents
from src.chunker import chunk_docs
from src.vectorstore import build_index

docs = load_documents()
chunks = chunk_docs(docs)
build_index(chunks)

print("Index built successfully")
