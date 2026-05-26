from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from torch import nn
from tqdm import tqdm


class TfidfSvdEncoder:
    def __init__(self, *, max_features: int = 12000, dense_dim: int = 256) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(n_components=dense_dim, random_state=20260526)

    def fit(self, texts: list[str]) -> None:
        matrix = self.vectorizer.fit_transform(texts)
        n_features = matrix.shape[1]
        if self.svd.n_components >= n_features:
            self.svd.n_components = max(1, n_features - 1)
        self.svd.fit(matrix)

    def encode(self, texts: list[str], *, batch_size: int = 256) -> np.ndarray:
        del batch_size
        matrix = self.vectorizer.transform(texts)
        dense = self.svd.transform(matrix).astype("float32")
        return normalize(dense, norm="l2").astype("float32")


class JinaEncoder:
    def __init__(self, *, model_name: str, device: str = "cpu") -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def fit(self, texts: list[str]) -> None:
        del texts

    @torch.no_grad()
    def encode(self, texts: list[str], *, batch_size: int = 64) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy().astype("float32"))
        return np.concatenate(vectors, axis=0)


class JinaSequenceEncoder:
    def __init__(
        self,
        *,
        model_name: str,
        device: str = "cpu",
        max_length: int = 128,
        cache_dir: Path | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model_name = model_name
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.tokenizer: Any | None = None
        self.model: Any | None = None

    def fit(self, texts: list[str]) -> None:
        del texts

    @torch.no_grad()
    def encode_sequence(
        self,
        texts: list[str],
        *,
        batch_size: int = 16,
        cache_key: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        cache_path = self._cache_path(texts=texts, cache_key=cache_key)
        if cache_path is not None and cache_path.exists():
            cached = np.load(cache_path)
            print(f"[cache] hit {cache_path}")
            return cached["sequences"], cached["masks"].astype("bool")

        self._ensure_loaded()
        sequences: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for start in tqdm(
            range(0, len(texts), batch_size),
            desc=f"jina_sequence_encode:{cache_key or 'texts'}",
            unit="batch",
        ):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**encoded)
            hidden = torch.nn.functional.normalize(outputs.last_hidden_state.float(), p=2, dim=-1)
            sequences.append(hidden.cpu().numpy().astype("float32"))
            masks.append(encoded["attention_mask"].cpu().numpy().astype("bool"))
        sequence_array = np.concatenate(sequences, axis=0)
        mask_array = np.concatenate(masks, axis=0)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, sequences=sequence_array, masks=mask_array.astype("uint8"))
            print(f"[cache] wrote {cache_path}")
        return sequence_array, mask_array

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def _cache_path(self, *, texts: list[str], cache_key: str | None) -> Path | None:
        if self.cache_dir is None:
            return None
        payload = {
            "cache_key": cache_key,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "texts": texts,
        }
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        safe_key = (cache_key or "texts").replace("/", "_").replace(" ", "_")
        return self.cache_dir / f"{safe_key}.{digest}.npz"


def build_encoder(config: dict[str, Any]) -> TfidfSvdEncoder | JinaEncoder:
    encoder_cfg = config["encoder"]
    backend = encoder_cfg.get("backend", "tfidf")
    if backend == "tfidf":
        return TfidfSvdEncoder(
            max_features=int(encoder_cfg.get("max_features", 12000)),
            dense_dim=int(encoder_cfg.get("dense_dim", 256)),
        )
    if backend == "jina":
        return JinaEncoder(
            model_name=str(encoder_cfg["model_name"]),
            device=str(config.get("train", {}).get("device", "cpu")),
        )
    raise ValueError(f"unknown encoder backend: {backend}")


def build_sequence_encoder(config: dict[str, Any]) -> JinaSequenceEncoder:
    encoder_cfg = config["encoder"]
    backend = encoder_cfg.get("backend", "jina_sequence")
    if backend != "jina_sequence":
        raise ValueError(f"sequence mode requires encoder.backend='jina_sequence', got {backend}")
    return JinaSequenceEncoder(
        model_name=str(encoder_cfg["model_name"]),
        device=str(config.get("train", {}).get("device", "cpu")),
        max_length=int(encoder_cfg.get("max_length", 128)),
        cache_dir=Path(str(encoder_cfg["cache_dir"])) if encoder_cfg.get("cache_dir") else None,
    )


def build_pair_features(conv_vectors: np.ndarray, tool_vectors: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            conv_vectors,
            tool_vectors,
            np.abs(conv_vectors - tool_vectors),
            conv_vectors * tool_vectors,
        ],
        axis=1,
    ).astype("float32")


@dataclass(frozen=True)
class EncodedPairs:
    features: np.ndarray
    labels: np.ndarray
    conversation_ids: list[str]
    tool_ids: list[str]
    raw_scores: np.ndarray


@dataclass(frozen=True)
class EncodedSequencePairs:
    conv_sequences: np.ndarray
    conv_masks: np.ndarray
    tool_sequences: np.ndarray
    tool_masks: np.ndarray
    labels: np.ndarray
    conversation_ids: list[str]
    tool_ids: list[str]
    raw_scores: np.ndarray


class PairMLPRegressor(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class LateInteractionRegressor(nn.Module):
    def __init__(self, *, hidden_size: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 4 + 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        conv_seq: torch.Tensor,
        conv_mask: torch.Tensor,
        tool_seq: torch.Tensor,
        tool_mask: torch.Tensor,
    ) -> torch.Tensor:
        conv_seq = torch.nn.functional.normalize(conv_seq, p=2, dim=-1)
        tool_seq = torch.nn.functional.normalize(tool_seq, p=2, dim=-1)
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
        features = torch.cat(
            [
                conv_pool,
                tool_pool,
                torch.abs(conv_pool - tool_pool),
                conv_pool * tool_pool,
                stats,
            ],
            dim=1,
        )
        return self.head(features).squeeze(-1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)


def _masked_scalar_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _masked_topk_mean(values: torch.Tensor, mask: torch.Tensor, *, k: int) -> torch.Tensor:
    masked = values.masked_fill(~mask, -1e4)
    topk = masked.topk(k=min(k, masked.shape[1]), dim=1).values
    valid = topk > -1e3
    return (topk.masked_fill(~valid, 0.0)).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
