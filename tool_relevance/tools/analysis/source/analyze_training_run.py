from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bad-cases", type=int, default=20)
    args = parser.parse_args()

    metrics_path = args.run_dir / "metrics.jsonl"
    report_path = args.run_dir / "report.json"
    predictions_path = args.run_dir / "val_predictions.jsonl"

    metrics = [json.loads(line) for line in metrics_path.open(encoding="utf-8") if line.strip()]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print("epochs", len(metrics))
    print("best_epoch", report["best"]["epoch"])
    print("best_val", json.dumps(report["best"]["val"], indent=2, sort_keys=True))

    rows = [json.loads(line) for line in predictions_path.open(encoding="utf-8") if line.strip()]
    bad = sorted(rows, key=lambda row: abs(row["prediction"] - row["label"]), reverse=True)
    print("\nworst prediction errors")
    for row in bad[: args.bad_cases]:
        print(
            f"{row['conversation_id']} {row['tool_id']} "
            f"label={row['label']:.3f} pred={row['prediction']:.3f} raw={row['raw_score']:.1f}"
        )


if __name__ == "__main__":
    main()
