from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from data import PairExample, load_dataset
from metrics import append_jsonl, compute_metrics, write_json
from model import EncodedPairs, PairMLPRegressor, build_encoder, build_pair_features


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

    encoder = build_encoder(config)
    fit_texts = sorted(
        {example.conversation_text for example in bundle.train_examples}
        | {example.tool_text for example in bundle.train_examples}
    )
    print(f"[encoder] backend={config['encoder']['backend']} fit_texts={len(fit_texts)}")
    encoder.fit(fit_texts)

    train_pairs = encode_examples(bundle.train_examples, encoder, config)
    val_pairs = encode_examples(bundle.val_examples, encoder, config)

    device = torch.device(config["train"].get("device", "cpu"))
    model = PairMLPRegressor(
        input_dim=train_pairs.features.shape[1],
        hidden_dim=int(config["model"].get("hidden_dim", 256)),
        dropout=float(config["model"].get("dropout", 0.1)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"].get("lr", 1e-3)),
        weight_decay=float(config["train"].get("weight_decay", 1e-4)),
    )
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_pairs.features),
            torch.from_numpy(train_pairs.labels),
        ),
        batch_size=int(config["train"].get("batch_size", 256)),
        shuffle=True,
    )

    best_score = -1.0
    best_report: dict[str, Any] = {}
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    for epoch in range(1, int(config["train"].get("epochs", 30)) + 1):
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            full_train=train_pairs,
            config=config,
            device=device,
        )
        train_metrics = evaluate(model, train_pairs, config, device)
        val_metrics = evaluate(model, val_pairs, config, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train": train_metrics,
            "val": val_metrics,
        }
        append_jsonl(metrics_path, row)
        print(
            f"[epoch {epoch:03d}] loss={train_loss:.4f} "
            f"val_mae={val_metrics['mae']:.4f} "
            f"val_top1={val_metrics['top1_match']:.3f} "
            f"val_top3={val_metrics['topk_recall']:.3f} "
            f"val_no_tool_fp={val_metrics['no_tool_fp_rate']:.3f}"
        )
        score = val_metrics["topk_recall"] - val_metrics["no_tool_fp_rate"] - val_metrics["mae"]
        if score > best_score:
            best_score = score
            best_report = row
            torch.save(model.state_dict(), run_dir / "best.pt")

    torch.save(model.state_dict(), run_dir / "last.pt")
    write_predictions(run_dir / "val_predictions.jsonl", model, val_pairs, device)
    write_json(run_dir / "report.json", {"best_score": best_score, "best": best_report})
    print(f"[done] run_dir={run_dir}")


def encode_examples(examples: list[PairExample], encoder: Any, config: dict[str, Any]) -> EncodedPairs:
    batch_size = int(config["encoder"].get("batch_size", 64))
    conv_vectors = encoder.encode([example.conversation_text for example in examples], batch_size=batch_size)
    tool_vectors = encoder.encode([example.tool_text for example in examples], batch_size=batch_size)
    return EncodedPairs(
        features=build_pair_features(conv_vectors, tool_vectors),
        labels=np.asarray([example.label for example in examples], dtype="float32"),
        conversation_ids=[example.conversation_id for example in examples],
        tool_ids=[example.tool_id for example in examples],
        raw_scores=np.asarray([example.raw_score for example in examples], dtype="float32"),
    )


def train_one_epoch(
    *,
    model: PairMLPRegressor,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    full_train: EncodedPairs,
    config: dict[str, Any],
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    criterion = torch.nn.SmoothL1Loss()
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        predictions = model(features)
        loss = criterion(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    pairwise_weight = float(config["train"].get("pairwise_weight", 0.0))
    if pairwise_weight > 0:
        # One lightweight full-set ranking step per epoch; simple and easy to inspect.
        features = torch.from_numpy(full_train.features).to(device)
        labels = torch.from_numpy(full_train.labels).to(device)
        predictions = model(features)
        rank_loss = pairwise_margin_loss(
            predictions=predictions,
            labels=labels,
            conversation_ids=full_train.conversation_ids,
            margin=float(config["train"].get("pairwise_margin", 0.1)),
        )
        optimizer.zero_grad()
        (pairwise_weight * rank_loss).backward()
        optimizer.step()
        losses.append(float(rank_loss.detach().cpu()))
    return float(sum(losses) / max(1, len(losses)))


def pairwise_margin_loss(
    *,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    conversation_ids: list[str],
    margin: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    by_group: dict[str, list[int]] = {}
    for index, conversation_id in enumerate(conversation_ids):
        by_group.setdefault(conversation_id, []).append(index)
    for indices in by_group.values():
        if len(indices) < 2:
            continue
        idx = torch.tensor(indices, device=predictions.device)
        group_predictions = predictions[idx]
        group_labels = labels[idx]
        best = int(torch.argmax(group_labels))
        worst = int(torch.argmin(group_labels))
        if float(group_labels[best] - group_labels[worst]) < 0.05:
            continue
        losses.append(torch.relu(margin - (group_predictions[best] - group_predictions[worst])))
    if not losses:
        return predictions.sum() * 0.0
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(
    model: PairMLPRegressor,
    pairs: EncodedPairs,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    features = torch.from_numpy(pairs.features).to(device)
    predictions = model(features).cpu().numpy()
    eval_cfg = config["eval"]
    return compute_metrics(
        labels=pairs.labels,
        predictions=predictions,
        conversation_ids=pairs.conversation_ids,
        high_label_threshold=float(eval_cfg.get("high_label_threshold", 0.8)),
        high_pred_threshold=float(eval_cfg.get("high_pred_threshold", 0.8)),
        topk=int(eval_cfg.get("topk", 3)),
    )


@torch.no_grad()
def write_predictions(
    path: Path,
    model: PairMLPRegressor,
    pairs: EncodedPairs,
    device: torch.device,
) -> None:
    model.eval()
    predictions = model(torch.from_numpy(pairs.features).to(device)).cpu().numpy()
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
