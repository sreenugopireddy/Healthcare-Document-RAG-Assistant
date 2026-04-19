"""
run_benchmark.py
────────────────
Phase 2 entry point. Run from repo root:
    python run_benchmark.py

Takes ~8 minutes total (6 configs × ~65s rate-limit gap between each).
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    sys.exit("❌  GROQ_API_KEY not set in .env")

from rag_eval.benchmark_runner import run_benchmark

if __name__ == "__main__":
    print("\n🔬 Starting Phase 2 — Multi-Config RAG Benchmark")
    print("   6 configs × ~65s gap = ~8 minutes total\n")
    run_benchmark()