from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from metrics import compute_metrics, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (args.run_dir / "val_predictions.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    metrics = compute_metrics(
        labels=np.asarray([row["label"] for row in rows], dtype="float32"),
        predictions=np.asarray([row["prediction"] for row in rows], dtype="float32"),
        conversation_ids=[row["conversation_id"] for row in rows],
        tool_ids=[row["tool_id"] for row in rows],
        high_label_threshold=float(config["eval"].get("high_label_threshold", 0.8)),
        high_pred_threshold=float(config["eval"].get("high_pred_threshold", 0.8)),
        topk=int(config["eval"].get("topk", 3)),
    )
    write_json(args.run_dir / "metrics_recomputed.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
