"""
rag_eval/evaluator.py
"""

import json
import time
from pathlib import Path
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


def _load_benchmark(path):
    with open(path) as f:
        return json.load(f)


def _run_rag_on_questions(rag_chain, retriever, questions):
    rows = []
    total = len(questions)

    for i, item in enumerate(questions, 1):
        q            = item["question"]
        ground_truth = item["ground_truth"]

        print(f"  [{i}/{total}] {q[:70]}...")
        t0 = time.time()

        # call your rag function
        result = rag_chain(q)
        answer = result.get("answer", "")

        # get contexts for RAGAS
        docs     = retriever.invoke(q)
        contexts = [doc.page_content for doc in docs]

        latency = round(time.time() - t0, 2)
        print(f"         ✓ {latency}s  |  {answer[:60]}...")

        rows.append({
            "question":     q,
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": ground_truth,
            "latency_s":    latency,
        })

    return rows


def run_evaluation(rag_chain: Any, retriever: Any, benchmark_path) -> dict:
    print("\n── Loading benchmark dataset ──")
    benchmark = _load_benchmark(benchmark_path)
    print(f"   {len(benchmark)} questions loaded")

    print("\n── Running RAG pipeline ──")
    rows = _run_rag_on_questions(rag_chain, retriever, benchmark)

    ragas_data = {
        "question":     [r["question"]     for r in rows],
        "answer":       [r["answer"]       for r in rows],
        "contexts":     [r["contexts"]     for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    }
    dataset = Dataset.from_dict(ragas_data)

    print("\n── Running RAGAS scoring ──")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextRecall(llm=llm),
    ]

    ragas_result = evaluate(dataset, metrics=metrics)
    scores_df    = ragas_result.to_pandas()

    # per-question results
    per_question = []
    for i, row in enumerate(rows):
        per_question.append({
            "question":         row["question"],
            "answer":           row["answer"],
            "latency_s":        row["latency_s"],
            "faithfulness":     round(float(scores_df.iloc[i].get("faithfulness",     0)), 3),
            "answer_relevancy": round(float(scores_df.iloc[i].get("answer_relevancy", 0)), 3),
            "context_recall":   round(float(scores_df.iloc[i].get("context_recall",   0)), 3),
        })

    def avg(col):
        return round(float(scores_df[col].mean()), 3) if col in scores_df.columns else 0.0

    aggregate = {
        "faithfulness":     avg("faithfulness"),
        "answer_relevancy": avg("answer_relevancy"),
        "context_recall":   avg("context_recall"),
        "avg_latency_s":    round(sum(r["latency_s"] for r in rows) / len(rows), 2),
        "n_questions":      len(rows),
    }

    print("\n── Aggregate Scores ──")
    for k, v in aggregate.items():
        print(f"   {k:<22} {v}")

    return {"per_question": per_question, "aggregate": aggregate}