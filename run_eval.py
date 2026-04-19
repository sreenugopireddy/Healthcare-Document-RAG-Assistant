"""
run_eval.py
───────────
Run from the root of Healthcare-Document-RAG-Assistant:
    python run_eval.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    sys.exit("❌  GROQ_API_KEY not set in .env")

from src.vectorstore import load_or_create_index
from src.rag_chain import build_rag

sys.path.insert(0, str(Path(__file__).parent))
from rag_eval import run_evaluation, save_results

BENCHMARK_PATH = Path(__file__).parent / "rag_eval" / "data" / "benchmark_qa.json"


def main():
    print("=" * 55)
    print("  Healthcare RAG Evaluation — Phase 1")
    print("=" * 55)

    print("\n── Loading vector store ──")
    db        = load_or_create_index()
    retriever = db.as_retriever(search_kwargs={"k": 3})
    rag       = build_rag(retriever)
    print("   RAG pipeline ready ✓")

    results = run_evaluation(
        rag_chain      = rag,
        retriever      = retriever,
        benchmark_path = BENCHMARK_PATH,
    )

    save_results(results, config_name="baseline_k3_llama3")

    print("\n── Per-Question Scores ──────────────────────────────────────────")
    print(f"{'#':<3} {'Question':<45} {'Faith':>6} {'Relev':>6} {'Recall':>7} {'ms':>6}")
    print("-" * 75)
    for i, r in enumerate(results["per_question"], 1):
        print(
            f"{i:<3} {r['question'][:44]:<45} "
            f"{r['faithfulness']:>6.2f} "
            f"{r['answer_relevancy']:>6.2f} "
            f"{r['context_recall']:>7.2f} "
            f"{int(r['latency_s']*1000):>6}"
        )
    print("-" * 75)
    agg = results["aggregate"]
    print(
        f"{'AVG':<3} {'':<45} "
        f"{agg['faithfulness']:>6.2f} "
        f"{agg['answer_relevancy']:>6.2f} "
        f"{agg['context_recall']:>7.2f} "
        f"{int(agg['avg_latency_s']*1000):>6}"
    )
    print("=" * 75)
    print(f"\n✅  Evaluation complete.")
    print(f"    Faithfulness:      {agg['faithfulness']:.3f}")
    print(f"    Answer Relevancy:  {agg['answer_relevancy']:.3f}")
    print(f"    Context Recall:    {agg['context_recall']:.3f}")
    print(f"    Avg Latency:       {agg['avg_latency_s']}s\n")


if __name__ == "__main__":
    main()