from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

def load_documents(path="data"):
    docs = []
    for pdf in Path(path).glob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        loaded = loader.load()
        for d in loaded:
            d.metadata["source_file"] = pdf.name
        docs.extend(loaded)
    return docs
