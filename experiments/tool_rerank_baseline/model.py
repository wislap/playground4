from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from torch import nn


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
