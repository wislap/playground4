from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import sparse
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


class LexicalFeatureBuilder:
    def __init__(
        self,
        *,
        word_max_features: int = 20000,
        char_max_features: int = 20000,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=word_max_features,
            lowercase=True,
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=char_max_features,
            lowercase=True,
            sublinear_tf=True,
        )
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.avg_doc_len = 1.0
        self.doc_freq: dict[str, int] = {}
        self.doc_count = 0

    def fit(self, conversation_texts: list[str], tool_texts: list[str]) -> None:
        texts = conversation_texts + tool_texts
        self.word_vectorizer.fit(texts)
        self.char_vectorizer.fit(texts)

        tool_token_sets = []
        doc_lengths = []
        for text in tool_texts:
            tokens = _lex_tokens(text)
            doc_lengths.append(len(tokens))
            tool_token_sets.append(set(tokens))
        self.doc_count = len(tool_texts)
        self.avg_doc_len = float(sum(doc_lengths) / max(1, len(doc_lengths)))
        self.doc_freq = {}
        for token_set in tool_token_sets:
            for token in token_set:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    def transform(self, conversation_texts: list[str], tool_texts: list[str]) -> np.ndarray:
        word_conv = self.word_vectorizer.transform(conversation_texts)
        word_tool = self.word_vectorizer.transform(tool_texts)
        char_conv = self.char_vectorizer.transform(conversation_texts)
        char_tool = self.char_vectorizer.transform(tool_texts)
        word_cos = _row_cosine(word_conv, word_tool)
        char_cos = _row_cosine(char_conv, char_tool)

        rows: list[list[float]] = []
        for conv_text, tool_text, word_score, char_score in zip(
            conversation_texts,
            tool_texts,
            word_cos,
            char_cos,
            strict=True,
        ):
            conv_tokens = _lex_tokens(conv_text)
            tool_tokens = _lex_tokens(tool_text)
            conv_set = set(conv_tokens)
            tool_set = set(tool_tokens)
            overlap = conv_set & tool_set
            union = conv_set | tool_set
            tool_id_tokens = set(_tool_id_tokens(tool_text))
            bm25 = self._bm25(query_tokens=conv_tokens, doc_tokens=tool_tokens)
            rows.append(
                [
                    float(word_score),
                    float(char_score),
                    float(bm25),
                    len(overlap) / max(1, len(conv_set)),
                    len(overlap) / max(1, len(tool_set)),
                    len(overlap) / max(1, len(union)),
                    len(conv_set & tool_id_tokens) / max(1, len(tool_id_tokens)),
                    1.0 if conv_set & tool_id_tokens else 0.0,
                ]
            )
        features = np.asarray(rows, dtype="float32")
        features[:, 2] = np.log1p(features[:, 2])
        return features

    @property
    def feature_dim(self) -> int:
        return 8

    def _bm25(self, *, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        tf: dict[str, int] = {}
        for token in doc_tokens:
            tf[token] = tf.get(token, 0) + 1
        doc_len = len(doc_tokens)
        score = 0.0
        for token in set(query_tokens):
            freq = tf.get(token, 0)
            if not freq:
                continue
            df = self.doc_freq.get(token, 0)
            idf = np.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
            denom = freq + self.bm25_k1 * (
                1.0 - self.bm25_b + self.bm25_b * doc_len / max(1.0, self.avg_doc_len)
            )
            score += idf * (freq * (self.bm25_k1 + 1.0)) / denom
        return float(score)


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


def build_lexical_features(config: dict[str, Any]) -> LexicalFeatureBuilder | None:
    lexical_cfg = config.get("lexical", {})
    if not lexical_cfg.get("enabled", False):
        return None
    return LexicalFeatureBuilder(
        word_max_features=int(lexical_cfg.get("word_max_features", 20000)),
        char_max_features=int(lexical_cfg.get("char_max_features", 20000)),
        bm25_k1=float(lexical_cfg.get("bm25_k1", 1.5)),
        bm25_b=float(lexical_cfg.get("bm25_b", 0.75)),
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
    lexical_features: np.ndarray | None
    policy_features: np.ndarray | None
    gate_features: np.ndarray | None
    labels: np.ndarray
    axis_labels: np.ndarray | None
    conversation_ids: list[str]
    tool_ids: list[str]
    raw_scores: np.ndarray


@dataclass(frozen=True)
class EncodedFieldSequencePairs:
    field_names: list[str]
    conv_sequences: np.ndarray
    conv_masks: np.ndarray
    field_sequences: dict[str, np.ndarray]
    field_masks: dict[str, np.ndarray]
    lexical_features: np.ndarray | None
    labels: np.ndarray
    conversation_ids: list[str]
    tool_ids: list[str]
    raw_scores: np.ndarray


class PairMLPRegressor(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        head: str = "mlp",
    ) -> None:
        super().__init__()
        self.net = _build_head(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            head=head,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class LateInteractionRegressor(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        lexical_dim: int = 0,
        lexical_fusion: str = "concat",
        lexical_dropout: float = 0.0,
        policy_dim: int = 0,
        policy_scale: float = 0.3,
        policy_dropout: float = 0.0,
        policy_hidden_dim: int = 32,
        gate_dim: int = 0,
        gate_dropout: float = 0.0,
        gate_hidden_dim: int = 16,
        conv_bias_scale: float = 0.0,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        head: str = "mlp",
    ) -> None:
        super().__init__()
        semantic_dim = hidden_size * 4 + 6
        self.lexical_fusion = lexical_fusion
        self.lexical_dropout = nn.Dropout(lexical_dropout)
        if lexical_dim > 0 and lexical_fusion == "gated_add":
            self.lexical_projection = nn.Linear(lexical_dim, semantic_dim)
            self.lexical_gate = nn.Sequential(
                nn.Linear(semantic_dim + lexical_dim, max(16, hidden_dim // 2)),
                nn.GELU(),
                nn.Linear(max(16, hidden_dim // 2), 1),
                nn.Sigmoid(),
            )
            head_input_dim = semantic_dim
        elif lexical_dim > 0 and lexical_fusion == "gated_concat":
            self.lexical_projection = None
            self.lexical_gate = nn.Sequential(
                nn.Linear(semantic_dim + lexical_dim, max(16, hidden_dim // 2)),
                nn.GELU(),
                nn.Linear(max(16, hidden_dim // 2), lexical_dim),
                nn.Sigmoid(),
            )
            head_input_dim = semantic_dim + lexical_dim
        elif lexical_dim > 0 and lexical_fusion == "concat":
            self.lexical_projection = None
            self.lexical_gate = None
            head_input_dim = semantic_dim + lexical_dim
        elif lexical_dim == 0:
            self.lexical_projection = None
            self.lexical_gate = None
            head_input_dim = semantic_dim
        else:
            raise ValueError(f"unknown lexical_fusion: {lexical_fusion}")
        self.head = _build_head(
            input_dim=head_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            head=head,
        )
        self.policy_scale = float(policy_scale)
        self.conv_bias_scale = float(conv_bias_scale)
        self.policy_dropout = nn.Dropout(policy_dropout)
        self.gate_dropout = nn.Dropout(gate_dropout)
        self.policy_residual = (
            _build_small_head(input_dim=policy_dim, hidden_dim=policy_hidden_dim, dropout=policy_dropout)
            if policy_dim > 0 and self.policy_scale > 0.0
            else None
        )
        self.policy_gate = (
            _build_small_head(input_dim=gate_dim, hidden_dim=gate_hidden_dim, dropout=gate_dropout)
            if gate_dim > 0 and self.policy_residual is not None
            else None
        )
        self.conv_bias = (
            _build_small_head(input_dim=gate_dim, hidden_dim=gate_hidden_dim, dropout=gate_dropout)
            if gate_dim > 0 and self.conv_bias_scale > 0.0
            else None
        )

    def build_features(
        self,
        conv_seq: torch.Tensor,
        conv_mask: torch.Tensor,
        tool_seq: torch.Tensor,
        tool_mask: torch.Tensor,
        lexical_features: torch.Tensor | None = None,
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
        semantic_features = torch.cat(
            [conv_pool, tool_pool, torch.abs(conv_pool - tool_pool), conv_pool * tool_pool, stats],
            dim=1,
        )
        features = semantic_features
        if lexical_features is not None:
            lexical_input = self.lexical_dropout(lexical_features)
            if self.lexical_fusion == "gated_add":
                if self.lexical_projection is None or self.lexical_gate is None:
                    raise RuntimeError("gated lexical fusion is not initialized")
                gate = self.lexical_gate(torch.cat([semantic_features, lexical_input], dim=1))
                features = semantic_features + gate * self.lexical_projection(lexical_input)
            elif self.lexical_fusion == "gated_concat":
                if self.lexical_gate is None:
                    raise RuntimeError("gated lexical fusion is not initialized")
                gate = self.lexical_gate(torch.cat([semantic_features, lexical_input], dim=1))
                features = torch.cat([semantic_features, gate * lexical_input], dim=1)
            else:
                features = torch.cat([semantic_features, lexical_input], dim=1)
        return features

    def forward(
        self,
        conv_seq: torch.Tensor,
        conv_mask: torch.Tensor,
        tool_seq: torch.Tensor,
        tool_mask: torch.Tensor,
        lexical_features: torch.Tensor | None = None,
        policy_features: torch.Tensor | None = None,
        gate_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.build_features(
            conv_seq=conv_seq,
            conv_mask=conv_mask,
            tool_seq=tool_seq,
            tool_mask=tool_mask,
            lexical_features=lexical_features,
        )
        score = self.head(features).squeeze(-1)
        if policy_features is not None and self.policy_residual is not None:
            policy_input = self.policy_dropout(policy_features)
            policy_delta = self.policy_scale * torch.tanh(self.policy_residual(policy_input).squeeze(-1))
            if gate_features is not None and self.policy_gate is not None:
                gate_input = self.gate_dropout(gate_features)
                policy_multiplier = 2.0 * torch.sigmoid(self.policy_gate(gate_input).squeeze(-1))
                policy_delta = policy_multiplier * policy_delta
            score = score + policy_delta
        if gate_features is not None and self.conv_bias is not None:
            gate_input = self.gate_dropout(gate_features)
            score = score + self.conv_bias_scale * torch.tanh(self.conv_bias(gate_input).squeeze(-1))
        return score


class FactorizedLateInteractionRegressor(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        factor_count: int,
        lexical_dim: int = 0,
        lexical_fusion: str = "concat",
        lexical_dropout: float = 0.0,
        policy_dim: int = 0,
        policy_scale: float = 0.3,
        policy_dropout: float = 0.0,
        policy_hidden_dim: int = 32,
        gate_dim: int = 0,
        gate_dropout: float = 0.0,
        gate_hidden_dim: int = 16,
        conv_bias_scale: float = 0.0,
        hidden_dim: int = 256,
        factor_hidden_dim: int = 128,
        final_feature_dim: int = 32,
        dropout: float = 0.1,
        final_head: str = "linear",
        use_factor_interactions: bool = False,
        detach_factors_for_final: bool = False,
        final_gate: bool = False,
        final_gate_hidden_dim: int = 32,
        final_gate_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.factor_count = factor_count
        self.use_factor_interactions = use_factor_interactions
        self.detach_factors_for_final = detach_factors_for_final
        self.final_gate_scale = final_gate_scale
        self.base = LateInteractionRegressor(
            hidden_size=hidden_size,
            lexical_dim=lexical_dim,
            lexical_fusion=lexical_fusion,
            lexical_dropout=lexical_dropout,
            policy_dim=policy_dim,
            policy_scale=policy_scale,
            policy_dropout=policy_dropout,
            policy_hidden_dim=policy_hidden_dim,
            gate_dim=gate_dim,
            gate_dropout=gate_dropout,
            gate_hidden_dim=gate_hidden_dim,
            conv_bias_scale=conv_bias_scale,
            hidden_dim=hidden_dim,
            dropout=dropout,
            head="linear",
        )
        feature_dim = int(self.base.head.in_features)
        factor_hidden_dim = max(8, factor_hidden_dim)
        self.factor_head = nn.Sequential(
            nn.Linear(feature_dim, factor_hidden_dim),
            nn.LayerNorm(factor_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(factor_hidden_dim, factor_count),
        )
        self.final_feature_projection = (
            nn.Sequential(
                nn.Linear(feature_dim, final_feature_dim),
                nn.LayerNorm(final_feature_dim),
                nn.GELU(),
            )
            if final_feature_dim > 0
            else None
        )
        interaction_dim = 7 if use_factor_interactions else 0
        final_input_dim = factor_count + interaction_dim + max(0, final_feature_dim)
        self.final_head = _build_head(
            input_dim=final_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            head=final_head,
        )
        self.final_gate = (
            _build_small_head(
                input_dim=final_input_dim,
                hidden_dim=final_gate_hidden_dim,
                dropout=dropout,
            )
            if final_gate and final_gate_scale > 0.0
            else None
        )

    def forward(
        self,
        conv_seq: torch.Tensor,
        conv_mask: torch.Tensor,
        tool_seq: torch.Tensor,
        tool_mask: torch.Tensor,
        lexical_features: torch.Tensor | None = None,
        policy_features: torch.Tensor | None = None,
        gate_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.base.build_features(
            conv_seq=conv_seq,
            conv_mask=conv_mask,
            tool_seq=tool_seq,
            tool_mask=tool_mask,
            lexical_features=lexical_features,
        )
        factors = self.factor_head(features)
        final_factors = factors.detach() if self.detach_factors_for_final else factors
        final_parts = [final_factors]
        if self.use_factor_interactions:
            final_parts.append(_factor_interactions(final_factors))
        if self.final_feature_projection is not None:
            final_parts.append(self.final_feature_projection(features))
        final_input = torch.cat(final_parts, dim=1)
        final = self.final_head(final_input).squeeze(-1)
        if self.final_gate is not None:
            final = final - self.final_gate_scale * torch.sigmoid(self.final_gate(final_input).squeeze(-1))
        if policy_features is not None and self.base.policy_residual is not None:
            policy_input = self.base.policy_dropout(policy_features)
            policy_delta = self.base.policy_scale * torch.tanh(self.base.policy_residual(policy_input).squeeze(-1))
            if gate_features is not None and self.base.policy_gate is not None:
                gate_input = self.base.gate_dropout(gate_features)
                policy_multiplier = 2.0 * torch.sigmoid(self.base.policy_gate(gate_input).squeeze(-1))
                policy_delta = policy_multiplier * policy_delta
            final = final + policy_delta
        if gate_features is not None and self.base.conv_bias is not None:
            gate_input = self.base.gate_dropout(gate_features)
            final = final + self.base.conv_bias_scale * torch.tanh(self.base.conv_bias(gate_input).squeeze(-1))
        return {"final": final, "factors": factors}


class FieldInteractionRegressor(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        field_names: list[str],
        lexical_dim: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        head: str = "mlp",
    ) -> None:
        super().__init__()
        self.field_names = field_names
        field_feature_dim = hidden_size * 4 + 6
        input_dim = field_feature_dim * len(field_names) + lexical_dim
        self.head = _build_head(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            head=head,
        )

    def forward(
        self,
        conv_seq: torch.Tensor,
        conv_mask: torch.Tensor,
        field_sequences: dict[str, torch.Tensor],
        field_masks: dict[str, torch.Tensor],
        lexical_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = []
        conv_seq = torch.nn.functional.normalize(conv_seq, p=2, dim=-1)
        for field_name in self.field_names:
            parts.append(
                _late_interaction_features(
                    conv_seq=conv_seq,
                    conv_mask=conv_mask,
                    tool_seq=field_sequences[field_name],
                    tool_mask=field_masks[field_name],
                )
            )
        if lexical_features is not None:
            parts.append(lexical_features)
        return self.head(torch.cat(parts, dim=1)).squeeze(-1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)


def _build_head(*, input_dim: int, hidden_dim: int, dropout: float, head: str) -> nn.Module:
    if head == "linear":
        return nn.Linear(input_dim, 1)
    if head == "mlp":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    if head == "swiglu":
        return SwiGLUHead(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout, expansion=2.67)
    raise ValueError(f"unknown model.head: {head}")


class SwiGLUHead(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, dropout: float, expansion: float = 2.67) -> None:
        super().__init__()
        inner_dim = max(8, round(hidden_dim * expansion))
        output_hidden_dim = max(8, hidden_dim // 2)
        self.up_gate = nn.Linear(input_dim, inner_dim * 2)
        self.norm = nn.LayerNorm(inner_dim)
        self.dropout = nn.Dropout(dropout)
        self.down = nn.Linear(inner_dim, output_hidden_dim)
        self.out = nn.Linear(output_hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values, gates = self.up_gate(features).chunk(2, dim=-1)
        hidden = values * torch.nn.functional.silu(gates)
        hidden = self.norm(hidden)
        hidden = self.dropout(hidden)
        hidden = torch.nn.functional.gelu(self.down(hidden))
        return self.out(hidden)


def _build_small_head(*, input_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
    hidden_dim = max(1, hidden_dim)
    if hidden_dim == 1:
        return nn.Linear(input_dim, 1)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


def _factor_interactions(factors: torch.Tensor) -> torch.Tensor:
    if factors.shape[1] < 6:
        raise ValueError("factor interactions require six factor predictions")
    capability = factors[:, 0]
    action = factors[:, 1]
    specificity = factors[:, 2]
    consent = factors[:, 3]
    cost = factors[:, 4]
    style = factors[:, 5]
    return torch.stack(
        [
            capability * action,
            capability * specificity,
            action * consent,
            consent * style,
            action * style,
            cost * action,
            cost * consent,
        ],
        dim=1,
    )


def _masked_scalar_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _masked_topk_mean(values: torch.Tensor, mask: torch.Tensor, *, k: int) -> torch.Tensor:
    masked = values.masked_fill(~mask, -1e4)
    topk = masked.topk(k=min(k, masked.shape[1]), dim=1).values
    valid = topk > -1e3
    return (topk.masked_fill(~valid, 0.0)).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def _late_interaction_features(
    *,
    conv_seq: torch.Tensor,
    conv_mask: torch.Tensor,
    tool_seq: torch.Tensor,
    tool_mask: torch.Tensor,
) -> torch.Tensor:
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
    return torch.cat([conv_pool, tool_pool, torch.abs(conv_pool - tool_pool), conv_pool * tool_pool, stats], dim=1)


def _lex_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())


def _tool_id_tokens(tool_text: str) -> list[str]:
    first_line = tool_text.splitlines()[0] if tool_text else ""
    if first_line.startswith("tool_id:"):
        tool_id = first_line.split(":", 1)[1]
        return [token for token in re.split(r"[^a-zA-Z0-9]+", tool_id.lower()) if token]
    return []


def _row_cosine(left: sparse.spmatrix, right: sparse.spmatrix) -> np.ndarray:
    return np.asarray(left.multiply(right).sum(axis=1)).ravel().astype("float32")
