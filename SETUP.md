# Phase 1 Setup — RAG Evaluation Engine

## What this adds to your existing repo

```
Healthcare-Document-RAG-Assistant/
├── app.py                          ← unchanged
├── build_index.py                  ← unchanged
├── src/                            ← unchanged
│
├── run_eval.py                     ← NEW: entry point
├── requirements_eval.txt           ← NEW: eval dependencies
├── rag_eval/                       ← NEW: eval module
│   ├── __init__.py
│   ├── evaluator.py                ← RAGAS scoring engine
│   ├── logger.py                   ← saves JSON + CSV results
│   └── data/
│       └── benchmark_qa.json       ← 10 healthcare QA pairs
└── results/                        ← auto-created on first run
    ├── latest.json
    ├── run_baseline_k3_gpt35_<ts>.json
    └── summary.csv
```

---

## Step 1 — Copy files into your repo

Copy these into the root of Healthcare-Document-RAG-Assistant:
- run_eval.py
- requirements_eval.txt
- rag_eval/ (entire folder)

---

## Step 2 — Install eval dependencies

```bash
pip install -r requirements_eval.txt
```

---

## Step 3 — Confirm your .env has OpenAI key

```
OPENAI_API_KEY=sk-...
```

(You likely already have this since app.py requires it)

---

## Step 4 — Build your FAISS index (if not already built)

```bash
python build_index.py
```

---

## Step 5 — Run the evaluation

```bash
python run_eval.py
```

Expected output:
```
=======================================================
  Healthcare RAG Evaluation — Phase 1
=======================================================

── Loading vector store ──
   Vector store loaded ✓

── Loading benchmark dataset ──
   10 questions loaded

── Running RAG pipeline ──
  [1/10] What are the common symptoms of type 2 diabetes?...
  ...

── Running RAGAS evaluation ──
  ...

── Per-Question Scores ──────────────────────────────────────────────
#   Question                                      Faith  Relev  Recall    ms
---------------------------------------------------------------------------
1   What are the common symptoms of type 2 d...   0.92   0.87    0.78   1823
...
AVG                                                0.85   0.82    0.74   1950
=============================================================================

✅  Baseline eval complete.
    Faithfulness:      0.850
    Answer Relevancy:  0.820
    Context Recall:    0.740
    Avg Latency:       1.95s

    Results saved to results/
```

---

## What these scores mean for your resume

| Score | Meaning |
|-------|---------|
| Faithfulness | Is the answer grounded in retrieved docs? (1.0 = perfect) |
| Answer Relevancy | Does the answer actually address the question? |
| Context Recall | Did retrieval surface the right information? |

**Resume bullet template:**
> "Evaluated RAG pipeline using RAGAS framework across 10 benchmark queries; achieved faithfulness score of X.XX, answer relevancy X.XX, context recall X.XX on healthcare QA dataset"

---

## Next steps (Phase 2)

Phase 2 will add a benchmark runner that tests multiple configs:
- chunk_size: [256, 512, 1024]
- chunk_overlap: [0, 50, 100]
- top_k: [3, 5, 7]

All tracked in MLflow. This is what produces the leaderboard.
