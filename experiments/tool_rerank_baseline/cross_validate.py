from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from data import ConversationGroup, bundle_from_groups, load_all_groups
from metrics import append_jsonl, compute_metrics, write_json
from model import LateInteractionRegressor, build_lexical_features, build_sequence_encoder
from train import (
    build_tensor_dataset,
    encode_sequence_examples,
    predict_all,
    set_seed,
    train_one_epoch,
    write_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    base_config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    set_seed(int(base_config["data"].get("seed", 20260526)))

    cv_cfg = base_config["cross_validation"]
    run_dir = Path(base_config["output"]["run_root"]) / base_config["output"]["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / "config.resolved.toml")

    all_groups = load_all_groups(base_config)
    base_train, cv_groups, holdout_groups = split_for_cv(
        all_groups,
        cv_ratio=float(cv_cfg.get("cv_ratio", 0.2)),
        holdout_ratio=float(cv_cfg.get("holdout_ratio", 0.2)),
        seed=int(base_config["data"].get("seed", 20260526)),
    )
    folds = make_folds(cv_groups, fold_count=int(cv_cfg.get("folds", 5)))
    print(
        f"[cv] total_groups={len(all_groups)} base_train={len(base_train)} "
        f"cv_groups={len(cv_groups)} holdout={len(holdout_groups)} folds={len(folds)}"
    )

    device = torch.device(base_config["train"].get("device", "cpu"))
    fold_reports = []
    for fold_index, fold_val in enumerate(folds, start=1):
        fold_train = base_train + [group for i, fold in enumerate(folds, start=1) if i != fold_index for group in fold]
        fold_config = deepcopy(base_config)
        fold_config["output"]["run_name"] = f"{base_config['output']['run_name']}/fold_{fold_index:02d}"
        fold_dir = run_dir / f"fold_{fold_index:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[fold {fold_index}] train_groups={len(fold_train)} "
            f"val_groups={len(fold_val)} train_pairs={sum(len(g.examples) for g in fold_train)} "
            f"val_pairs={sum(len(g.examples) for g in fold_val)}"
        )
        report_path = fold_dir / "report.json"
        if report_path.exists():
            print(f"[fold {fold_index}] resume: using existing {report_path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = run_fold(
                config=fold_config,
                train_groups=fold_train,
                val_groups=fold_val,
                run_dir=fold_dir,
                device=device,
            )
        fold_reports.append({"fold": fold_index, **report})

    aggregate = aggregate_reports(fold_reports)
    write_json(
        run_dir / "cv_report.json",
        {
            "folds": fold_reports,
            "aggregate": aggregate,
            "split": {
                "base_train_groups": len(base_train),
                "cv_groups": len(cv_groups),
                "holdout_groups": len(holdout_groups),
            },
            "holdout_group_ids": [group.conversation_id for group in holdout_groups],
        },
    )
    print("[cv] aggregate")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    print(f"[done] run_dir={run_dir}")


def run_fold(
    *,
    config: dict[str, Any],
    train_groups: list[ConversationGroup],
    val_groups: list[ConversationGroup],
    run_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    bundle = bundle_from_groups(train_groups=train_groups, val_groups=val_groups)
    fit_texts = sorted(
        {example.conversation_text for example in bundle.train_examples}
        | {example.tool_text for example in bundle.train_examples}
    )
    encoder = build_sequence_encoder(config)
    encoder.fit(fit_texts)
    lexical_builder = build_lexical_features(config)
    lexical_dim = 0
    if lexical_builder is not None:
        lexical_builder.fit(
            sorted({example.conversation_text for example in bundle.train_examples}),
            sorted({example.tool_text for example in bundle.train_examples}),
        )
        lexical_dim = lexical_builder.feature_dim

    split_prefix = f"cv_fold_{run_dir.name}"
    train_pairs = encode_sequence_examples(
        bundle.train_examples,
        encoder,
        config,
        split_name=f"{split_prefix}_train",
        lexical_builder=lexical_builder,
    )
    val_pairs = encode_sequence_examples(
        bundle.val_examples,
        encoder,
        config,
        split_name=f"{split_prefix}_val",
        lexical_builder=lexical_builder,
    )
    model = LateInteractionRegressor(
        hidden_size=train_pairs.conv_sequences.shape[-1],
        lexical_dim=lexical_dim,
        lexical_fusion=str(config.get("lexical", {}).get("fusion", "concat")),
        lexical_dropout=float(config.get("lexical", {}).get("dropout", 0.0)),
        hidden_dim=int(config["model"].get("hidden_dim", 256)),
        dropout=float(config["model"].get("dropout", 0.1)),
        head=str(config["model"].get("head", "mlp")),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"].get("lr", 1e-3)),
        weight_decay=float(config["train"].get("weight_decay", 1e-4)),
    )
    train_loader = DataLoader(
        build_tensor_dataset(train_pairs),
        batch_size=int(config["train"].get("batch_size", 256)),
        shuffle=True,
    )

    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    best_score = -1.0
    best_report: dict[str, Any] = {}
    for epoch in range(1, int(config["train"].get("epochs", 30)) + 1):
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            full_train=train_pairs,
            config=config,
            device=device,
        )
        train_metrics = _evaluate_predictions(train_pairs, predict_all(model, train_pairs, device), config)
        val_metrics = _evaluate_predictions(val_pairs, predict_all(model, val_pairs, device), config)
        row = {"epoch": epoch, "train_loss": train_loss, "train": train_metrics, "val": val_metrics}
        append_jsonl(metrics_path, row)
        print(
            f"[{run_dir.name} epoch {epoch:03d}] loss={train_loss:.4f} "
            f"val_mae={val_metrics['mae']:.4f} val_top1={val_metrics['top1_match']:.3f} "
            f"val_top3={val_metrics['topk_recall']:.3f}"
        )
        score = val_metrics["topk_recall"] - val_metrics["no_tool_fp_rate"] - val_metrics["mae"]
        if score > best_score:
            best_score = score
            best_report = row
            torch.save(model.state_dict(), run_dir / "best.pt")

    torch.save(model.state_dict(), run_dir / "last.pt")
    write_predictions(run_dir / "val_predictions.jsonl", model, val_pairs, device)
    report = {"best_score": best_score, "best": best_report}
    write_json(run_dir / "report.json", report)
    return report


def split_for_cv(
    groups: list[ConversationGroup],
    *,
    cv_ratio: float,
    holdout_ratio: float,
    seed: int,
) -> tuple[list[ConversationGroup], list[ConversationGroup], list[ConversationGroup]]:
    if cv_ratio <= 0 or holdout_ratio <= 0 or cv_ratio + holdout_ratio >= 1:
        raise ValueError("cv_ratio and holdout_ratio must be positive and sum to less than 1")
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    holdout_count = round(len(shuffled) * holdout_ratio)
    cv_count = round(len(shuffled) * cv_ratio)
    holdout = shuffled[:holdout_count]
    cv_groups = shuffled[holdout_count : holdout_count + cv_count]
    base_train = shuffled[holdout_count + cv_count :]
    return base_train, cv_groups, holdout


def make_folds(groups: list[ConversationGroup], *, fold_count: int) -> list[list[ConversationGroup]]:
    if fold_count < 2:
        raise ValueError("folds must be >= 2")
    folds = [[] for _ in range(fold_count)]
    for index, group in enumerate(groups):
        folds[index % fold_count].append(group)
    return folds


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "mae",
        "mse",
        "spearman",
        "top1_match",
        "topk_recall",
        "no_tool_fp_rate",
        "high_conf_precision",
    ]
    aggregate = {}
    for key in keys:
        values = np.asarray([report["best"]["val"][key] for report in reports], dtype="float64")
        aggregate[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return aggregate


def _evaluate_predictions(pairs: Any, predictions: torch.Tensor, config: dict[str, Any]) -> dict[str, float]:
    eval_cfg = config["eval"]
    return compute_metrics(
        labels=pairs.labels,
        predictions=predictions.detach().cpu().numpy(),
        conversation_ids=pairs.conversation_ids,
        high_label_threshold=float(eval_cfg.get("high_label_threshold", 0.8)),
        high_pred_threshold=float(eval_cfg.get("high_pred_threshold", 0.8)),
        topk=int(eval_cfg.get("topk", 3)),
    )


if __name__ == "__main__":
    main()
