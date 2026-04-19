"""
rag_eval/benchmark_runner.py

Phase 2 — Multi-config benchmark runner with MLflow tracking.

Tests combinations of chunk_size, chunk_overlap, and top_k.
Logs every config as an MLflow run. Produces a leaderboard CSV.
"""

import time
import mlflow
import pandas as pd
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.loader import load_documents
from src.rag_chain import build_rag
from rag_eval.evaluator import run_evaluation
from rag_eval.logger import save_results

BENCHMARK_PATH = Path(__file__).parent / "data" / "benchmark_qa.json"
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
MLFLOW_EXP     = "healthcare-rag-benchmark"

# ── configs to test ──────────────────────────────────────────────────────────
CONFIGS = [
    {"chunk_size": 400,  "chunk_overlap": 50,  "top_k": 3},
    {"chunk_size": 400,  "chunk_overlap": 50,  "top_k": 5},
    {"chunk_size": 800,  "chunk_overlap": 150, "top_k": 3},  # your current baseline
    {"chunk_size": 800,  "chunk_overlap": 150, "top_k": 5},
    {"chunk_size": 1200, "chunk_overlap": 200, "top_k": 3},
    {"chunk_size": 1200, "chunk_overlap": 200, "top_k": 5},
]


def _build_retriever(chunk_size: int, chunk_overlap: int, top_k: int):
    """Rebuild FAISS index from scratch for this config."""
    print(f"   Building index: chunk_size={chunk_size}, overlap={chunk_overlap}...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )

    docs     = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    db        = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": top_k})

    print(f"   Index built: {len(chunks)} chunks")
    return retriever


def run_benchmark():
    mlflow.set_experiment(MLFLOW_EXP)

    leaderboard = []

    for i, cfg in enumerate(CONFIGS, 1):
        config_name = f"cs{cfg['chunk_size']}_co{cfg['chunk_overlap']}_k{cfg['top_k']}"
        print(f"\n{'='*60}")
        print(f"  Config {i}/{len(CONFIGS)}: {config_name}")
        print(f"{'='*60}")

        with mlflow.start_run(run_name=config_name):

            # log config params
            mlflow.log_params(cfg)

            # build retriever for this config
            retriever = _build_retriever(**cfg)
            rag       = build_rag(retriever)

            # run evaluation (with rate-limit delay between configs)
            results = run_evaluation(
                rag_chain      = rag,
                retriever      = retriever,
                benchmark_path = BENCHMARK_PATH,
            )

            agg = results["aggregate"]

            # log metrics to MLflow
            mlflow.log_metric("faithfulness",     agg["faithfulness"])
            mlflow.log_metric("answer_relevancy", agg["answer_relevancy"])
            mlflow.log_metric("context_recall",   agg["context_recall"])
            mlflow.log_metric("avg_latency_s",    agg["avg_latency_s"])

            # save full results to JSON
            save_results(results, config_name=config_name)

            # collect for leaderboard
            leaderboard.append({
                "config":           config_name,
                "chunk_size":       cfg["chunk_size"],
                "chunk_overlap":    cfg["chunk_overlap"],
                "top_k":            cfg["top_k"],
                "faithfulness":     agg["faithfulness"],
                "answer_relevancy": agg["answer_relevancy"],
                "context_recall":   agg["context_recall"],
                "avg_latency_s":    agg["avg_latency_s"],
            })

            print(f"\n   ✓ faithfulness={agg['faithfulness']}  "
                  f"relevancy={agg['answer_relevancy']}  "
                  f"recall={agg['context_recall']}  "
                  f"latency={agg['avg_latency_s']}s")

            # wait 65s between configs to reset Groq TPM limit
            if i < len(CONFIGS):
                print(f"\n   ⏳ Waiting 65s for Groq rate limit reset before next config...")
                time.sleep(65)

    # ── write leaderboard ──────────────────────────────────────────────────
    df = pd.DataFrame(leaderboard)
    df["composite_score"] = (
        df["faithfulness"] * 0.4 +
        df["answer_relevancy"] * 0.35 +
        df["context_recall"] * 0.25
    ).round(3)

    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.index += 1  # rank from 1

    leaderboard_path = Path("results") / "leaderboard.csv"
    df.to_csv(leaderboard_path)

    print("\n" + "="*60)
    print("  LEADERBOARD")
    print("="*60)
    print(df[["config", "faithfulness", "answer_relevancy",
              "context_recall", "avg_latency_s", "composite_score"]].to_string())
    print(f"\n✅  Leaderboard saved → {leaderboard_path}")
    print(f"    Best config: {df.iloc[0]['config']}")
    print(f"\n    Run `mlflow ui` to explore all experiments in browser.")

    return df