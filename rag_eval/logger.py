"""
rag_eval/logger.py

Saves evaluation results to:
  - results/latest.json        (full detail)
  - results/run_<timestamp>.json
  - results/summary.csv        (appends each run for trend tracking)
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path


RESULTS_DIR = Path(__file__).parent.parent / "results"


def save_results(results: dict, config_name: str = "default") -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{config_name}_{timestamp}"

    payload = {
        "run_id":    run_id,
        "timestamp": timestamp,
        "config":    config_name,
        **results,
    }

    # full run file
    run_path = RESULTS_DIR / f"run_{run_id}.json"
    with open(run_path, "w") as f:
        json.dump(payload, f, indent=2)

    # overwrite latest
    with open(RESULTS_DIR / "latest.json", "w") as f:
        json.dump(payload, f, indent=2)

    # append to summary CSV
    csv_path = RESULTS_DIR / "summary.csv"
    write_header = not csv_path.exists()
    agg = results["aggregate"]

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run_id", "timestamp", "config",
                "faithfulness", "answer_relevancy", "context_recall",
                "avg_latency_s", "n_questions",
            ])
        writer.writerow([
            run_id, timestamp, config_name,
            agg["faithfulness"], agg["answer_relevancy"], agg["context_recall"],
            agg["avg_latency_s"], agg["n_questions"],
        ])

    print(f"Results saved → {run_path}")
    print(f"Summary CSV  → {csv_path}")
    return run_path
