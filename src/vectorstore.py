from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "medical_index"


# Create embedding model ONCE at module load
_embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"}
)


def build_index(chunks):

    db = FAISS.from_documents(chunks, _embeddings)

    db.save_local(INDEX_PATH)

    print("✅ Vector index built and saved.")


def load_index():

    db = FAISS.load_local(
        INDEX_PATH,
        _embeddings,
        allow_dangerous_deserialization=True
    )

    return db
