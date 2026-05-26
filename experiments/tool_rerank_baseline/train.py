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
from model import (
    EncodedFieldSequencePairs,
    EncodedPairs,
    EncodedSequencePairs,
    FieldInteractionRegressor,
    LexicalFeatureBuilder,
    LateInteractionRegressor,
    PairMLPRegressor,
    build_encoder,
    build_lexical_features,
    build_pair_features,
    build_sequence_encoder,
)


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

    device = torch.device(config["train"].get("device", "cpu"))
    model_kind = config.get("model", {}).get("kind", "pair_mlp")
    fit_texts = sorted(
        {example.conversation_text for example in bundle.train_examples}
        | {example.tool_text for example in bundle.train_examples}
    )
    print(f"[encoder] backend={config['encoder']['backend']} fit_texts={len(fit_texts)}")
    if model_kind in {"sequence_late_interaction", "field_interaction"}:
        encoder = build_sequence_encoder(config)
        encoder.fit(fit_texts)
        lexical_builder = build_lexical_features(config)
        lexical_dim = 0
        if lexical_builder is not None:
            fit_conversations = sorted({example.conversation_text for example in bundle.train_examples})
            fit_tools = sorted({example.tool_text for example in bundle.train_examples})
            lexical_builder.fit(fit_conversations, fit_tools)
            lexical_dim = lexical_builder.feature_dim
            print(f"[lexical] enabled feature_dim={lexical_dim}")
        if model_kind == "field_interaction":
            field_names = list(config["model"].get("field_names", _default_field_names()))
            train_pairs = encode_field_sequence_examples(
                bundle.train_examples,
                encoder,
                config,
                split_name="train",
                field_names=field_names,
                lexical_builder=lexical_builder,
            )
            val_pairs = encode_field_sequence_examples(
                bundle.val_examples,
                encoder,
                config,
                split_name="val",
                field_names=field_names,
                lexical_builder=lexical_builder,
            )
            model = FieldInteractionRegressor(
                hidden_size=train_pairs.conv_sequences.shape[-1],
                field_names=field_names,
                lexical_dim=lexical_dim,
                hidden_dim=int(config["model"].get("hidden_dim", 256)),
                dropout=float(config["model"].get("dropout", 0.1)),
                head=str(config["model"].get("head", "mlp")),
            ).to(device)
        else:
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
            model = LateInteractionRegressor(
                hidden_size=train_pairs.conv_sequences.shape[-1],
                lexical_dim=lexical_dim,
                lexical_fusion=str(config.get("lexical", {}).get("fusion", "concat")),
                lexical_dropout=float(config.get("lexical", {}).get("dropout", 0.0)),
                hidden_dim=int(config["model"].get("hidden_dim", 256)),
                dropout=float(config["model"].get("dropout", 0.1)),
                head=str(config["model"].get("head", "mlp")),
            ).to(device)
    else:
        encoder = build_encoder(config)
        encoder.fit(fit_texts)
        train_pairs = encode_examples(bundle.train_examples, encoder, config)
        val_pairs = encode_examples(bundle.val_examples, encoder, config)
        model = PairMLPRegressor(
            input_dim=train_pairs.features.shape[1],
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
            f"val_ndcg5={val_metrics.get('ndcg_at_5', 0.0):.3f} "
            f"val_regret5={val_metrics.get('top5_regret', 0.0):.3f} "
            f"val_bad3={val_metrics.get('bad_top3_rate', 0.0):.3f} "
            f"val_no_tool_fp={val_metrics['no_tool_fp_rate']:.3f}"
        )
        score = val_metrics["topk_recall"] - val_metrics["no_tool_fp_rate"] - val_metrics["mae"]
        if score > best_score:
            best_score = score
            best_report = row
            torch.save(model.state_dict(), run_dir / "best.pt")
            write_predictions(run_dir / "best_val_predictions.jsonl", model, val_pairs, device)

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


def encode_sequence_examples(
    examples: list[PairExample],
    encoder: Any,
    config: dict[str, Any],
    *,
    split_name: str,
    lexical_builder: LexicalFeatureBuilder | None = None,
) -> EncodedSequencePairs:
    batch_size = int(config["encoder"].get("batch_size", 8))
    unique_conversations = sorted({example.conversation_id: example.conversation_text for example in examples}.items())
    unique_tools = sorted({example.tool_id: example.tool_text for example in examples}.items())
    conversation_ids = [item[0] for item in unique_conversations]
    tool_ids = [item[0] for item in unique_tools]
    conv_seq, conv_mask = encoder.encode_sequence(
        [item[1] for item in unique_conversations],
        batch_size=batch_size,
        cache_key=f"{split_name}_conversations",
    )
    tool_seq, tool_mask = encoder.encode_sequence(
        [item[1] for item in unique_tools],
        batch_size=batch_size,
        cache_key=f"{split_name}_tools",
    )
    conv_index = {conversation_id: index for index, conversation_id in enumerate(conversation_ids)}
    tool_index = {tool_id: index for index, tool_id in enumerate(tool_ids)}
    lexical_features = None
    if lexical_builder is not None:
        lexical_features = lexical_builder.transform(
            [example.conversation_text for example in examples],
            [example.tool_text for example in examples],
        )
    return EncodedSequencePairs(
        conv_sequences=np.asarray([conv_seq[conv_index[e.conversation_id]] for e in examples], dtype="float32"),
        conv_masks=np.asarray([conv_mask[conv_index[e.conversation_id]] for e in examples], dtype="bool"),
        tool_sequences=np.asarray([tool_seq[tool_index[e.tool_id]] for e in examples], dtype="float32"),
        tool_masks=np.asarray([tool_mask[tool_index[e.tool_id]] for e in examples], dtype="bool"),
        lexical_features=lexical_features,
        labels=np.asarray([example.label for example in examples], dtype="float32"),
        conversation_ids=[example.conversation_id for example in examples],
        tool_ids=[example.tool_id for example in examples],
        raw_scores=np.asarray([example.raw_score for example in examples], dtype="float32"),
    )


def encode_field_sequence_examples(
    examples: list[PairExample],
    encoder: Any,
    config: dict[str, Any],
    *,
    split_name: str,
    field_names: list[str],
    lexical_builder: LexicalFeatureBuilder | None = None,
) -> EncodedFieldSequencePairs:
    batch_size = int(config["encoder"].get("batch_size", 8))
    unique_conversations = sorted({example.conversation_id: example.conversation_text for example in examples}.items())
    unique_tools = sorted({example.tool_id: example.tool_fields for example in examples}.items())
    conversation_ids = [item[0] for item in unique_conversations]
    tool_ids = [item[0] for item in unique_tools]
    conv_seq, conv_mask = encoder.encode_sequence(
        [item[1] for item in unique_conversations],
        batch_size=batch_size,
        cache_key=f"{split_name}_conversations",
    )
    field_sequences = {}
    field_masks = {}
    for field_name in field_names:
        seq, mask = encoder.encode_sequence(
            [fields[field_name] for _, fields in unique_tools],
            batch_size=batch_size,
            cache_key=f"{split_name}_tools_{field_name}",
        )
        field_sequences[field_name] = seq
        field_masks[field_name] = mask

    conv_index = {conversation_id: index for index, conversation_id in enumerate(conversation_ids)}
    tool_index = {tool_id: index for index, tool_id in enumerate(tool_ids)}
    lexical_features = None
    if lexical_builder is not None:
        lexical_features = lexical_builder.transform(
            [example.conversation_text for example in examples],
            [example.tool_text for example in examples],
        )
    return EncodedFieldSequencePairs(
        field_names=field_names,
        conv_sequences=np.asarray([conv_seq[conv_index[e.conversation_id]] for e in examples], dtype="float32"),
        conv_masks=np.asarray([conv_mask[conv_index[e.conversation_id]] for e in examples], dtype="bool"),
        field_sequences={
            field_name: np.asarray(
                [field_sequences[field_name][tool_index[e.tool_id]] for e in examples],
                dtype="float32",
            )
            for field_name in field_names
        },
        field_masks={
            field_name: np.asarray(
                [field_masks[field_name][tool_index[e.tool_id]] for e in examples],
                dtype="bool",
            )
            for field_name in field_names
        },
        lexical_features=lexical_features,
        labels=np.asarray([example.label for example in examples], dtype="float32"),
        conversation_ids=[example.conversation_id for example in examples],
        tool_ids=[example.tool_id for example in examples],
        raw_scores=np.asarray([example.raw_score for example in examples], dtype="float32"),
    )


def build_tensor_dataset(pairs: EncodedPairs | EncodedSequencePairs | EncodedFieldSequencePairs) -> TensorDataset:
    if isinstance(pairs, EncodedFieldSequencePairs):
        tensors = [
            torch.from_numpy(pairs.conv_sequences),
            torch.from_numpy(pairs.conv_masks),
        ]
        for field_name in pairs.field_names:
            tensors.extend(
                [
                    torch.from_numpy(pairs.field_sequences[field_name]),
                    torch.from_numpy(pairs.field_masks[field_name]),
                ]
            )
        if pairs.lexical_features is not None:
            tensors.append(torch.from_numpy(pairs.lexical_features))
        tensors.append(torch.from_numpy(pairs.labels))
        return TensorDataset(*tensors)
    if isinstance(pairs, EncodedSequencePairs):
        if pairs.lexical_features is not None:
            return TensorDataset(
                torch.from_numpy(pairs.conv_sequences),
                torch.from_numpy(pairs.conv_masks),
                torch.from_numpy(pairs.tool_sequences),
                torch.from_numpy(pairs.tool_masks),
                torch.from_numpy(pairs.lexical_features),
                torch.from_numpy(pairs.labels),
            )
        return TensorDataset(
            torch.from_numpy(pairs.conv_sequences),
            torch.from_numpy(pairs.conv_masks),
            torch.from_numpy(pairs.tool_sequences),
            torch.from_numpy(pairs.tool_masks),
            torch.from_numpy(pairs.labels),
        )
    return TensorDataset(
        torch.from_numpy(pairs.features),
        torch.from_numpy(pairs.labels),
    )


def train_one_epoch(
    *,
    model: PairMLPRegressor | LateInteractionRegressor | FieldInteractionRegressor,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    full_train: EncodedPairs | EncodedSequencePairs | EncodedFieldSequencePairs,
    config: dict[str, Any],
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    criterion = torch.nn.SmoothL1Loss()
    for batch in loader:
        predictions, labels = predict_batch(model, batch, device)
        loss = criterion(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    pairwise_weight = float(config["train"].get("pairwise_weight", 0.0))
    if pairwise_weight > 0:
        # One lightweight full-set ranking step per epoch; simple and easy to inspect.
        labels = torch.from_numpy(full_train.labels).to(device)
        predictions = predict_all(model, full_train, device)
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


def predict_batch(
    model: PairMLPRegressor | LateInteractionRegressor | FieldInteractionRegressor,
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(batch) == 2:
        features, labels = batch
        return model(features.to(device)), labels.to(device)
    if len(batch) == 6:
        conv_seq, conv_mask, tool_seq, tool_mask, lexical_features, labels = batch
        return (
            model(
                conv_seq.to(device),
                conv_mask.to(device),
                tool_seq.to(device),
                tool_mask.to(device),
                lexical_features.to(device),
            ),
        labels.to(device),
    )
    if isinstance(model, FieldInteractionRegressor):
        conv_seq = batch[0]
        conv_mask = batch[1]
        offset = 2
        field_sequences = {}
        field_masks = {}
        for field_name in model.field_names:
            field_sequences[field_name] = batch[offset].to(device)
            field_masks[field_name] = batch[offset + 1].to(device)
            offset += 2
        lexical_features = None
        if len(batch) == 2 + len(model.field_names) * 2 + 2:
            lexical_features = batch[offset].to(device)
            offset += 1
        labels = batch[offset]
        return (
            model(
                conv_seq.to(device),
                conv_mask.to(device),
                field_sequences,
                field_masks,
                lexical_features,
            ),
            labels.to(device),
        )
    conv_seq, conv_mask, tool_seq, tool_mask, labels = batch
    return (
        model(
            conv_seq.to(device),
            conv_mask.to(device),
            tool_seq.to(device),
            tool_mask.to(device),
        ),
        labels.to(device),
    )


def predict_all(
    model: PairMLPRegressor | LateInteractionRegressor | FieldInteractionRegressor,
    pairs: EncodedPairs | EncodedSequencePairs | EncodedFieldSequencePairs,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(pairs, EncodedFieldSequencePairs):
        outputs = []
        batch_size = 256
        for start in range(0, len(pairs.labels), batch_size):
            end = min(start + batch_size, len(pairs.labels))
            lexical_features = (
                torch.from_numpy(pairs.lexical_features[start:end]).to(device)
                if pairs.lexical_features is not None
                else None
            )
            outputs.append(
                model(
                    torch.from_numpy(pairs.conv_sequences[start:end]).to(device),
                    torch.from_numpy(pairs.conv_masks[start:end]).to(device),
                    {
                        field_name: torch.from_numpy(pairs.field_sequences[field_name][start:end]).to(device)
                        for field_name in pairs.field_names
                    },
                    {
                        field_name: torch.from_numpy(pairs.field_masks[field_name][start:end]).to(device)
                        for field_name in pairs.field_names
                    },
                    lexical_features,
                )
            )
        return torch.cat(outputs, dim=0)
    if isinstance(pairs, EncodedSequencePairs):
        lexical_features = (
            torch.from_numpy(pairs.lexical_features).to(device)
            if pairs.lexical_features is not None
            else None
        )
        return model(
            torch.from_numpy(pairs.conv_sequences).to(device),
            torch.from_numpy(pairs.conv_masks).to(device),
            torch.from_numpy(pairs.tool_sequences).to(device),
            torch.from_numpy(pairs.tool_masks).to(device),
            lexical_features,
        )
    return model(torch.from_numpy(pairs.features).to(device))


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
    model: PairMLPRegressor | LateInteractionRegressor | FieldInteractionRegressor,
    pairs: EncodedPairs | EncodedSequencePairs | EncodedFieldSequencePairs,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions = predict_all(model, pairs, device).cpu().numpy()
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


@torch.no_grad()
def write_predictions(
    path: Path,
    model: PairMLPRegressor | LateInteractionRegressor | FieldInteractionRegressor,
    pairs: EncodedPairs | EncodedSequencePairs | EncodedFieldSequencePairs,
    device: torch.device,
) -> None:
    model.eval()
    predictions = predict_all(model, pairs, device).cpu().numpy()
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


def _default_field_names() -> list[str]:
    return ["identity", "description", "capabilities", "examples", "schema"]


if __name__ == "__main__":
    main()
