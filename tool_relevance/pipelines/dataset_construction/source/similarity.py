from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from itertools import combinations

from tool_relevance_lab.dataset_generation.schemas import ToolRecord, ToolUniverse


STOP_TOKENS = {
    "plugin",
    "assistant",
    "helper",
    "manager",
    "monitor",
    "bridge",
    "auto",
    "reply",
    "sync",
    "watcher",
    "agent",
    "工具",
    "助手",
    "管理",
    "监听",
    "自动",
    "回复",
    "同步",
    "提醒",
    "生成",
    "支持",
    "用户",
    "本地",
    "连接",
}


@dataclass(frozen=True)
class ToolSimilarityPair:
    left_tool_id: str
    right_tool_id: str
    score: float
    text_cosine: float
    keyword_jaccard: float
    name_similarity: float
    id_family_similarity: float
    left_name: str
    right_name: str
    left_short_description: str
    right_short_description: str

    def to_dict(self) -> dict:
        return asdict(self)


def find_similar_tools(
    universe: ToolUniverse,
    *,
    min_score: float = 0.28,
    min_text_cosine: float = 0.42,
    min_keyword_jaccard: float = 0.30,
    min_id_family_similarity: float = 0.50,
    plugins_only: bool = True,
) -> list[ToolSimilarityPair]:
    tools = universe.plugins if plugins_only else universe.tools
    vectors = {tool.tool_id: _token_counts(tool) for tool in tools}
    pairs: list[ToolSimilarityPair] = []
    for left, right in combinations(tools, 2):
        text_cosine = _cosine(vectors[left.tool_id], vectors[right.tool_id])
        keyword_jaccard = _keyword_jaccard(left, right)
        name_similarity = SequenceMatcher(None, _name(left), _name(right)).ratio()
        id_family_similarity = _id_family_similarity(left, right)
        score = (
            0.58 * text_cosine
            + 0.22 * keyword_jaccard
            + 0.10 * name_similarity
            + 0.10 * id_family_similarity
        )
        if (
            score >= min_score
            or text_cosine >= min_text_cosine
            or keyword_jaccard >= min_keyword_jaccard
            or id_family_similarity >= min_id_family_similarity
        ):
            pairs.append(
                ToolSimilarityPair(
                    left_tool_id=left.tool_id,
                    right_tool_id=right.tool_id,
                    score=round(score, 6),
                    text_cosine=round(text_cosine, 6),
                    keyword_jaccard=round(keyword_jaccard, 6),
                    name_similarity=round(name_similarity, 6),
                    id_family_similarity=round(id_family_similarity, 6),
                    left_name=_name(left),
                    right_name=_name(right),
                    left_short_description=str(left.metadata.get("short_description", "")),
                    right_short_description=str(right.metadata.get("short_description", "")),
                )
            )
    return sorted(pairs, key=lambda item: item.score, reverse=True)


def summarize_similarity(universe: ToolUniverse, pairs: list[ToolSimilarityPair]) -> dict:
    return {
        "tool_universe_id": universe.tool_universe_id,
        "tool_count": len(universe.tools),
        "plugin_count": len(universe.plugins),
        "similar_pair_count": len(pairs),
        "pairs": [pair.to_dict() for pair in pairs],
    }


def _token_counts(tool: ToolRecord) -> Counter[str]:
    text = " ".join(
        [
            tool.tool_id,
            str(tool.metadata.get("name", "")),
            str(tool.metadata.get("description", "")),
            str(tool.metadata.get("short_description", "")),
            " ".join(str(item) for item in tool.metadata.get("keywords", [])),
        ]
    )
    tokens = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]{1,2}", text.lower())
    return Counter(token for token in tokens if token not in STOP_TOKENS and len(token.strip("_")) > 1)


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    dot = sum(value * right.get(token, 0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _keyword_jaccard(left: ToolRecord, right: ToolRecord) -> float:
    left_keywords = {str(item).lower() for item in left.metadata.get("keywords", [])}
    right_keywords = {str(item).lower() for item in right.metadata.get("keywords", [])}
    if not left_keywords and not right_keywords:
        return 0.0
    return len(left_keywords & right_keywords) / len(left_keywords | right_keywords)


def _id_family_similarity(left: ToolRecord, right: ToolRecord) -> float:
    left_parts = set(left.tool_id.removeprefix("plugin.").split("_"))
    right_parts = set(right.tool_id.removeprefix("plugin.").split("_"))
    if not left_parts and not right_parts:
        return 0.0
    return len(left_parts & right_parts) / len(left_parts | right_parts)


def _name(tool: ToolRecord) -> str:
    return str(tool.metadata.get("name", ""))
