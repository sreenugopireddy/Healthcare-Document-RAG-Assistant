
Healthcare Document RAG Assistant

A domain-grounded Healthcare Question Answering assistant built using a Retrieval-Augmented Generation (RAG) pipeline.  
The system answers medical questions strictly from approved healthcare documents and provides source citations for every answer.

This project demonstrates an Agentic RAG workflow combining document retrieval, semantic search, and controlled LLM generation.

---

## 🎯 Objective

Large language models can hallucinate in medical contexts. This project solves that by:

- Retrieving relevant medical document passages
- Generating answers only from retrieved context
- Showing document source attribution
- Refusing out-of-scope queries

The result is a grounded, auditable healthcare knowledge assistant.

---

## 🧠 System Architecture
Medical PDFs
↓
Document Loader
↓
Text Chunking (overlap windows)
↓
SentenceTransformer Embeddings
↓
FAISS Vector Index
↓
Semantic Retrieval (Top-K)
↓
Context-Bound Prompt
↓
LLM Generation (Groq LLaMA)
↓
Answer + Source Citation


Core design rule: **No retrieval → No answer.**

---

# 📚 Knowledge Base

The assistant indexes public healthcare documents including:

- Pneumonia fact sheet
- Adult immunization schedule
- National immunization schedule
- Antibiotic resistance material
- Clinical preventive care guidelines

Only indexed documents are used for answering.

---

# ⚙️ Tech Stack

- Python
- LangChain (modular packages)
- FAISS vector database
- Sentence-Transformers embeddings
- Groq LLaMA-3.1 model
- Streamlit UI
- python-dotenv for secrets

---

# 📦 Project Structure



Healthcare-Document-RAG-Assistant/
│
├── data/ # Healthcare PDFs
├── medical_index/ # FAISS index (generated)
├── src/
│ ├── loader.py
│ ├── chunker.py
│ ├── vectorstore.py
│ ├── rag_chain.py
│ └── evaluator.py
│
├── app.py # Streamlit app
├── build_index.py # Vector index builder
├── evaluation_queries.json
├── requirements.txt
├── .env # API keys (not committed)
└── README.md


---

# 🔧 Installation

## 1️⃣ Create Virtual Environment



python -m venv venv
venv\Scripts\activate


## 2️⃣ Install Dependencies



pip install -r requirements.txt


---

# 🔐 API Setup (Groq — Free Tier)

Create Groq API key:

https://console.groq.com

Create `.env` file in project root:



GROQ_API_KEY=your_key_here


Do NOT commit `.env` to GitHub.

---

# 🏗️ Build Vector Index (Required First Run)

After placing PDFs in `/data`:



python build_index.py


This creates:



medical_index/
index.faiss
index.pkl


---

# ▶️ Run Application



streamlit run app.py


Open browser:



http://localhost:8501


---

# 🧪 Example Queries



What are the symptoms of pneumonia?
Do antibiotics work against viruses?
Who should receive pneumococcal vaccine?
How is pneumonia prevented?
What is antibiotic resistance?


---

# 🛡️ Guardrail Behavior

Out-of-scope questions return:



Not found in medical knowledge base


This prevents hallucinated medical advice.

---

# 📊 Evaluation Method

Tested across:

- pneumonia facts
- vaccination schedules
- antibiotic usage
- preventive care
- out-of-scope queries

Evaluation checks:

- Retrieval relevance
- Answer grounding
- Source citation presence
- Hallucination avoidance

---

# 🤖 Agentic Characteristics

This system demonstrates agentic RAG patterns:

- Tool use → retriever as knowledge tool
- Decision gating → answer only if context exists
- Grounded reasoning → generation depends on evidence
- Safety refusal → declines unsupported questions

---

# ⚠️ Medical Disclaimer

This system provides information from public medical documents only.  
It is **not a substitute for professional medical advice, diagnosis, or treatment.**

---

# 🚀 Future Improvements

- Hybrid search (keyword + vector)
- Confidence scoring
- Multi-hop retrieval
- Query rewriting agent
- Structured medical ontology linking
- Feedback learning loop

---





