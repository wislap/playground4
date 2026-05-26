from __future__ import annotations

import argparse
import json
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from data import load_dataset
from metrics import compute_metrics, write_json
from model import EncodedSequencePairs, build_lexical_features, build_sequence_encoder
from train import encode_sequence_examples, set_seed


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

    fit_texts = sorted(
        {example.conversation_text for example in bundle.train_examples}
        | {example.tool_text for example in bundle.train_examples}
    )
    encoder = build_sequence_encoder(config)
    encoder.fit(fit_texts)

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

    alphas = np.asarray(config.get("ridge", {}).get("alphas", _default_alphas()), dtype="float64")
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=alphas, scoring="neg_mean_absolute_error"),
    )
    model.fit(train_x, train_pairs.labels)
    predictions = model.predict(val_x).astype("float32")
    metrics = compute_metrics(
        labels=val_pairs.labels,
        predictions=predictions,
        conversation_ids=val_pairs.conversation_ids,
        tool_ids=val_pairs.tool_ids,
        high_label_threshold=float(config["eval"].get("high_label_threshold", 0.8)),
        high_pred_threshold=float(config["eval"].get("high_pred_threshold", 0.8)),
        topk=int(config["eval"].get("topk", 3)),
    )
    ridge = model.named_steps["ridgecv"]
    report = {
        "best_score": metrics["topk_recall"] - metrics["no_tool_fp_rate"] - metrics["mae"],
        "best": {
            "epoch": 1,
            "train_loss": None,
            "train": compute_metrics(
                labels=train_pairs.labels,
                predictions=model.predict(train_x).astype("float32"),
                conversation_ids=train_pairs.conversation_ids,
                tool_ids=train_pairs.tool_ids,
                high_label_threshold=float(config["eval"].get("high_label_threshold", 0.8)),
                high_pred_threshold=float(config["eval"].get("high_pred_threshold", 0.8)),
                topk=int(config["eval"].get("topk", 3)),
            ),
            "val": metrics,
        },
        "ridge": {"alpha": float(ridge.alpha_)},
    }
    write_json(run_dir / "report.json", report)
    write_metrics(run_dir / "metrics.jsonl", report)
    write_predictions(run_dir / "val_predictions.jsonl", val_pairs, predictions)
    print(f"[ridge] alpha={ridge.alpha_}")
    print(
        f"[done] val_mae={metrics['mae']:.4f} "
        f"val_top1={metrics['top1_match']:.3f} val_top3={metrics['topk_recall']:.3f} "
        f"run_dir={run_dir}"
    )


def build_late_interaction_matrix(pairs: EncodedSequencePairs, *, batch_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in tqdm(range(0, len(pairs.labels), batch_size), desc="ridge_features", unit="batch"):
        end = min(start + batch_size, len(pairs.labels))
        semantic = _semantic_features(
            torch.from_numpy(pairs.conv_sequences[start:end]),
            torch.from_numpy(pairs.conv_masks[start:end]),
            torch.from_numpy(pairs.tool_sequences[start:end]),
            torch.from_numpy(pairs.tool_masks[start:end]),
        ).numpy()
        if pairs.lexical_features is not None:
            semantic = np.concatenate([semantic, pairs.lexical_features[start:end]], axis=1)
        chunks.append(semantic.astype("float32"))
    return np.concatenate(chunks, axis=0)


def _semantic_features(
    conv_seq: torch.Tensor,
    conv_mask: torch.Tensor,
    tool_seq: torch.Tensor,
    tool_mask: torch.Tensor,
) -> torch.Tensor:
    conv_seq = torch.nn.functional.normalize(conv_seq.float(), p=2, dim=-1)
    tool_seq = torch.nn.functional.normalize(tool_seq.float(), p=2, dim=-1)
    conv_mask_f = conv_mask.float()
    tool_mask_f = tool_mask.float()

    conv_pool = _masked_mean(conv_seq, conv_mask_f)
    tool_pool = _masked_mean(tool_seq, tool_mask_f)
    sim = torch.matmul(conv_seq, tool_seq.transpose(1, 2))
    pair_mask = conv_mask.unsqueeze(2) & tool_mask.unsqueeze(1)
    sim = sim.masked_fill(~pair_mask, -1e4)
    conv_to_tool = sim.max(dim=2).values.masked_fill(~conv_mask, 0.0)
    tool_to_conv = sim.max(dim=1).values.masked_fill(~tool_mask, 0.0)
    stats = torch.stack(
        [
            _masked_scalar_mean(conv_to_tool, conv_mask_f),
            conv_to_tool.max(dim=1).values,
            _masked_topk_mean(conv_to_tool, conv_mask, k=5),
            _masked_scalar_mean(tool_to_conv, tool_mask_f),
            tool_to_conv.max(dim=1).values,
            _masked_topk_mean(tool_to_conv, tool_mask, k=5),
        ],
        dim=1,
    )
    return torch.cat([conv_pool, tool_pool, torch.abs(conv_pool - tool_pool), conv_pool * tool_pool, stats], dim=1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)


def _masked_scalar_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _masked_topk_mean(values: torch.Tensor, mask: torch.Tensor, *, k: int) -> torch.Tensor:
    masked = values.masked_fill(~mask, -1e4)
    topk = masked.topk(k=min(k, masked.shape[1]), dim=1).values
    valid = topk > -1e3
    return (topk.masked_fill(~valid, 0.0)).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def _default_alphas() -> list[float]:
    return [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def write_metrics(path: Path, report: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(report["best"], ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def write_predictions(path: Path, pairs: EncodedSequencePairs, predictions: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, pred in enumerate(predictions):
            handle.write(
                json.dumps(
                    {
                        "conversation_id": pairs.conversation_ids[index],
                        "tool_id": pairs.tool_ids[index],
                        "label": float(pairs.labels[index]),
                        "prediction": float(pred),
                        "raw_score": float(pairs.raw_scores[index]),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            handle.write("\n")


if __name__ == "__main__":
    main()
