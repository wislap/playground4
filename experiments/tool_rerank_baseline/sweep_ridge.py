from __future__ import annotations

import argparse
import json
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from data import load_dataset
from metrics import compute_metrics, write_json
from model import build_lexical_features, build_sequence_encoder
from train import encode_sequence_examples, set_seed
from train_ridge import build_late_interaction_matrix, write_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    set_seed(int(config["data"].get("seed", 20260526)))

    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / "config.resolved.toml")

    bundle = load_dataset(config)
    print(
        f"[data] train_groups={len(bundle.train_groups)} val_groups={len(bundle.val_groups)} "
        f"train_pairs={len(bundle.train_examples)} val_pairs={len(bundle.val_examples)}"
    )
    encoder = build_sequence_encoder(config)
    encoder.fit(
        sorted(
            {example.conversation_text for example in bundle.train_examples}
            | {example.tool_text for example in bundle.train_examples}
        )
    )
    lexical_builder = build_lexical_features(config)
    if lexical_builder is not None:
        lexical_builder.fit(
            sorted({example.conversation_text for example in bundle.train_examples}),
            sorted({example.tool_text for example in bundle.train_examples}),
        )
        print(f"[lexical] enabled feature_dim={lexical_builder.feature_dim}")

    train_pairs = encode_sequence_examples(
        bundle.train_examples,
        encoder,
        config,
        split_name="train",
        lexical_builder=lexical_builder,
    )
    val_pairs = encode_sequence_examples(
        bundle.val_examples,
        encoder,
        config,
        split_name="val",
        lexical_builder=lexical_builder,
    )
    batch_size = int(config.get("ridge", {}).get("feature_batch_size", 256))
    train_x = build_late_interaction_matrix(train_pairs, batch_size=batch_size)
    val_x = build_late_interaction_matrix(val_pairs, batch_size=batch_size)
    print(f"[features] train={train_x.shape} val={val_x.shape}")

    rows = []
    predictions_by_alpha: dict[float, np.ndarray] = {}
    for alpha in _alphas(config):
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=alpha, random_state=int(config["data"].get("seed", 20260526))),
        )
        model.fit(train_x, train_pairs.labels)
        val_predictions = model.predict(val_x).astype("float32")
        train_predictions = model.predict(train_x).astype("float32")
        val_metrics = _metrics(config, val_pairs, val_predictions)
        train_metrics = _metrics(config, train_pairs, train_predictions)
        row = {
            "alpha": alpha,
            "score": val_metrics["topk_recall"] - val_metrics["no_tool_fp_rate"] - val_metrics["mae"],
            "train": train_metrics,
            "val": val_metrics,
        }
        rows.append(row)
        predictions_by_alpha[alpha] = val_predictions
        print(
            f"[alpha {alpha:g}] val_mae={val_metrics['mae']:.4f} "
            f"top1={val_metrics['top1_match']:.3f} top3={val_metrics['topk_recall']:.3f} "
            f"spearman={val_metrics['spearman']:.3f} score={row['score']:.4f}"
        )

    best_by_score = max(rows, key=lambda row: row["score"])
    best_by_top3 = max(rows, key=lambda row: (row["val"]["topk_recall"], row["val"]["top1_match"], -row["val"]["mae"]))
    best_by_mae = min(rows, key=lambda row: row["val"]["mae"])
    report = {
        "rows": rows,
        "best_by_score": best_by_score,
        "best_by_top3": best_by_top3,
        "best_by_mae": best_by_mae,
    }
    write_json(run_dir / "ridge_sweep_report.json", report)
    write_json(
        run_dir / "report.json",
        {
            "best_score": best_by_score["score"],
            "best": {
                "epoch": 1,
                "train_loss": None,
                "train": best_by_score["train"],
                "val": best_by_score["val"],
            },
            "ridge": {"alpha": best_by_score["alpha"], "selection": "score"},
        },
    )
    write_metrics(run_dir / "metrics.jsonl", rows)
    write_predictions(
        run_dir / "val_predictions.jsonl",
        val_pairs,
        predictions_by_alpha[float(best_by_score["alpha"])],
    )
    print("[best_by_score]", _brief(best_by_score))
    print("[best_by_top3]", _brief(best_by_top3))
    print("[best_by_mae]", _brief(best_by_mae))
    print(f"[done] run_dir={run_dir}")


def _metrics(config: dict[str, Any], pairs: Any, predictions: np.ndarray) -> dict[str, float]:
    eval_cfg = config["eval"]
    return compute_metrics(
        labels=pairs.labels,
        predictions=predictions,
        conversation_ids=pairs.conversation_ids,
        tool_ids=pairs.tool_ids,
        high_label_threshold=float(eval_cfg.get("high_label_threshold", 0.8)),
        high_pred_threshold=float(eval_cfg.get("high_pred_threshold", 0.8)),
        topk=int(eval_cfg.get("topk", 3)),
    )


def _alphas(config: dict[str, Any]) -> list[float]:
    values = config.get("ridge", {}).get(
        "alphas",
        [
            0.0,
            1e-8,
            3e-8,
            1e-7,
            3e-7,
            1e-6,
            3e-6,
            1e-5,
            3e-5,
            1e-4,
            3e-4,
            1e-3,
            3e-3,
            1e-2,
            3e-2,
            0.1,
            0.3,
            1.0,
            3.0,
            10.0,
            30.0,
            100.0,
            300.0,
            1000.0,
            3000.0,
            10000.0,
        ],
    )
    return [float(value) for value in values]


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _brief(row: dict[str, Any]) -> str:
    val = row["val"]
    return (
        f"alpha={row['alpha']:g} score={row['score']:.4f} mae={val['mae']:.4f} "
        f"top1={val['top1_match']:.3f} top3={val['topk_recall']:.3f} "
        f"spearman={val['spearman']:.3f}"
    )


if __name__ == "__main__":
    main()
