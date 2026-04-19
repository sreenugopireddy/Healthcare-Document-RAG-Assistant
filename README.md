# Healthcare Document RAG Assistant

**A production-grade Retrieval-Augmented Generation system for medical knowledge QA — with a built-in evaluation framework that benchmarks retrieval quality across multiple pipeline configurations.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?style=flat-square)](https://langchain.com)
[![RAGAS](https://img.shields.io/badge/RAGAS-Evaluated-orange?style=flat-square)](https://ragas.io)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-red?style=flat-square)](https://mlflow.org)
[![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203-purple?style=flat-square)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## What makes this different

Most RAG projects stop at building the pipeline. This one **measures it**.

The system includes a complete evaluation framework that scores every RAG configuration on faithfulness, answer relevancy, and context recall using RAGAS — then logs all experiments to MLflow and visualises the leaderboard in a Streamlit dashboard.

| What most projects do | What this project does |
|---|---|
| Build a RAG system | Build + evaluate with RAGAS metrics |
| No quality measurement | Faithfulness 1.000 · Relevancy 0.736 · Recall 0.625 |
| Single configuration | Benchmark 6 configs across chunk size and top-k |
| No experiment tracking | MLflow logs every run for reproducible comparison |
| No visualisation | Streamlit leaderboard dashboard with Plotly charts |

---

## Evaluation Results (Baseline)

Evaluated on a curated 10-question benchmark aligned to the healthcare corpus.

```
Config: chunk_size=800, chunk_overlap=150, top_k=3
Model:  Groq LLaMA 3.1 8B (generation) + LLaMA 3.3 70B (judge)

┌─────────────────────┬────────┬────────────────────────────────────────────┐
│ Metric              │ Score  │ What it measures                           │
├─────────────────────┼────────┼────────────────────────────────────────────┤
│ Faithfulness        │ 1.000  │ Is the answer grounded in retrieved docs?  │
│ Answer Relevancy    │ 0.736  │ Does the answer address the question?      │
│ Context Recall      │ 0.625  │ Did retrieval surface the right context?   │
│ Avg Latency         │ 0.52s  │ End-to-end response time                   │
└─────────────────────┴────────┴────────────────────────────────────────────┘
```

**Faithfulness of 1.0** confirms guarded prompting eliminates hallucination.
**Context Recall of 0.625** identifies retrieval as the area to improve — exactly what Phase 2 benchmarking addresses.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OFFLINE INDEXING                             │
│                                                                 │
│  Healthcare PDFs → Loader → Chunker → Embeddings → FAISS Index │
│  (RecursiveCharacterTextSplitter, chunk=800, overlap=150)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ONLINE QUERY                                 │
│                                                                 │
│  User Query → Embed → Top-K FAISS Search → Context Assembly    │
│           → Guarded Prompt → Groq LLaMA → Answer + Sources     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION LAYER                             │
│                                                                 │
│  Benchmark QA → RAG Pipeline → RAGAS Scoring → MLflow Logging  │
│              → 6 Config Comparison → Streamlit Leaderboard     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | Core implementation |
| RAG Framework | LangChain | Pipeline orchestration |
| Vector DB | FAISS | Semantic similarity search |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Dense vector encoding |
| LLM | Groq LLaMA 3.1 8B Instant | Fast inference, zero cost |
| Evaluation | RAGAS 0.2+ | Faithfulness, relevancy, recall scoring |
| Judge LLM | Groq LLaMA 3.3 70B | RAGAS evaluation judge |
| Experiment Tracking | MLflow | Multi-config run comparison |
| Interface | Streamlit | Query UI + Leaderboard dashboard |
| Visualisation | Plotly | Interactive benchmark charts |

---

## Project Structure

```
Healthcare-Document-RAG-Assistant/
│
├── app.py                        # Streamlit query interface
├── build_index.py                # Offline FAISS index builder
├── run_eval.py                   # Phase 1: baseline evaluation
├── run_benchmark.py              # Phase 2: multi-config benchmark
├── eval_dashboard.py             # Phase 3: Streamlit leaderboard
├── requirements.txt              # Core dependencies
├── requirements_eval.txt         # Eval layer dependencies
│
├── src/
│   ├── loader.py                 # PDF ingestion
│   ├── chunker.py                # Text splitting strategy
│   ├── vectorstore.py            # FAISS index management
│   └── rag_chain.py              # RAG pipeline + guarded prompting
│
├── rag_eval/
│   ├── evaluator.py              # RAGAS scoring engine
│   ├── logger.py                 # JSON + CSV results persistence
│   ├── benchmark_runner.py       # Multi-config MLflow runner
│   └── data/
│       └── benchmark_qa.json     # 10 curated healthcare QA pairs
│
├── data/                         # Healthcare PDF corpus
└── results/                      # Evaluation outputs (gitignored)
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/sreenugopireddy/Healthcare-Document-RAG-Assistant.git
cd Healthcare-Document-RAG-Assistant

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Set environment variable

```bash
# .env
GROQ_API_KEY=your_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 3. Build the index

```bash
python build_index.py
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## RAG Evaluation Framework

### Phase 1 — Baseline Evaluation

Runs your RAG pipeline against 10 benchmark questions and scores with RAGAS.

```bash
pip install -r requirements_eval.txt
python run_eval.py
```

Output:
```
#   Question                                       Faith  Relev  Recall    ms
---------------------------------------------------------------------------
1   What is antibiotic resistance?                  1.00   1.00    0.50   890
2   How does antibiotic resistance spread?          1.00   1.00     --    470
3   Why are antibiotics ineffective vs viruses?     1.00   0.83    1.00   240
...
AVG                                                 1.00   0.74    0.62   520
```

### Phase 2 — Multi-Config Benchmark with MLflow

Tests 6 pipeline configurations automatically. Each run logged to MLflow.

```bash
python run_benchmark.py    # ~8 minutes (6 configs × 65s rate-limit gap)
mlflow ui                  # explore at http://localhost:5000
```

Configurations tested:

| Config | chunk_size | chunk_overlap | top_k |
|--------|-----------|---------------|-------|
| cs400_co50_k3  | 400  | 50  | 3 |
| cs400_co50_k5  | 400  | 50  | 5 |
| cs800_co150_k3 | 800  | 150 | 3 ← baseline |
| cs800_co150_k5 | 800  | 150 | 5 |
| cs1200_co200_k3 | 1200 | 200 | 3 |
| cs1200_co200_k5 | 1200 | 200 | 5 |

Composite score = faithfulness × 0.4 + relevancy × 0.35 + recall × 0.25

### Phase 3 — Streamlit Leaderboard Dashboard

```bash
streamlit run eval_dashboard.py    # http://localhost:8501
```

Features: grouped bar charts, composite score ranking, full results table, run history trend line.

---

## RAG Design Decisions

**Why guarded prompting?**
The prompt template explicitly instructs the model to answer only from retrieved context and respond "Not found in medical knowledge base" when evidence is absent. This is why faithfulness reaches 1.0 — the model never fabricates.

**Why FAISS over a hosted vector DB?**
For a local-first project the overhead of Pinecone or Weaviate adds no value. FAISS gives millisecond-latency similarity search without an API call, keeping end-to-end latency under 1 second.

**Why RAGAS for evaluation?**
RAGAS uses an LLM-as-judge approach to score faithfulness and relevancy without requiring human annotation. It's the standard for RAG evaluation and maps directly to the failure modes that matter — hallucination, off-topic answers, and retrieval gaps.

**Why Groq?**
Free tier, 300 RPM on LLaMA 3.1 8B, sub-second latency. The evaluation framework deliberately avoids OpenAI dependency to keep the project fully reproducible with a free Groq account.

---

## Prompt Engineering

The guarded prompt template enforces safe generation:

```
You are a healthcare assistant.

Answer ONLY using the provided context.
If the answer is not found, say:
"Not found in medical knowledge base."

Context:
{context}

Question:
{question}
```

This single constraint is responsible for the 1.0 faithfulness score. The model is never given permission to use its parametric knowledge.

---

## Publication

This project is accompanied by a technical article on RAG architecture for healthcare QA systems, published on Ready Tensor:

**Healthcare Document RAG Assistant: A Modular Retrieval-Augmented Generation System**
→ [Read on Ready Tensor](https://readytensor.ai)

---

## Future Work

- [ ] Hybrid BM25 + vector retrieval to improve context recall beyond 0.625
- [ ] Re-ranking layer (cross-encoder) for precision improvement
- [ ] Fine-tuned embedding model on medical terminology
- [ ] Larger corpus across ICD-10, clinical guidelines, drug references
- [ ] Automated nightly benchmark CI/CD pipeline
- [ ] Query expansion for recall improvement on short questions

---

## Disclaimer

This system is for educational and technical demonstration purposes only. It does not provide medical diagnosis, treatment advice, or clinical decision support.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Sreenivasa Reddy Gopireddy**
Data Science Student · RAG Systems · LLM Applications
[LinkedIn](https://linkedin.com/in/sreenugopireddy) · [GitHub](https://github.com/sreenugopireddy) · [Portfolio](https://sreenugopireddy.github.io)
