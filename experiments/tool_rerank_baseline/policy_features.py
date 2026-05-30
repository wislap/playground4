from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from data import PairExample


@dataclass(frozen=True)
class PolicyFeatureBuilder:
    enabled: bool = True
    groups: tuple[str, ...] = field(default_factory=lambda: ("conv", "tool", "interaction", "cross"))

    @property
    def feature_names(self) -> list[str]:
        return _feature_names_for_groups(self.groups)

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def transform(self, examples: list[PairExample]) -> np.ndarray:
        if not self.enabled:
            return np.zeros((len(examples), 0), dtype="float32")
        rows = []
        for example in examples:
            conv = _conversation_features(example)
            tool = _tool_features(example)
            rows.append(_assemble_features(example, conv, tool, groups=self.groups))
        return np.asarray(rows, dtype="float32")


CONV_FEATURES = [
    "conv_agentic_task",
    "conv_companion",
    "conv_boundary",
    "conv_proactive",
    "conv_open_thread",
    "conv_actionable",
    "conv_authorized",
    "conv_refusal",
    "conv_no_tool",
    "conv_no_reminder",
    "conv_no_upload",
    "conv_no_monitoring",
    "conv_no_search",
    "conv_no_file",
    "conv_no_planning",
    "conv_no_control",
    "conv_just_venting",
    "conv_quiet_presence",
]

TOOL_FEATURES = [
    "tool_agent",
    "tool_plugin",
    "tool_file",
    "tool_browser",
    "tool_search",
    "tool_message",
    "tool_calendar",
    "tool_reminder",
    "tool_monitoring",
    "tool_emotion",
    "tool_audio",
    "tool_camera",
    "tool_home",
    "tool_travel",
    "tool_code",
    "tool_research",
    "tool_study",
    "tool_social",
    "tool_external",
    "tool_proactive",
]

INTERACTION_FEATURES = [
    "conflict_refusal_domain",
    "conflict_no_reminder_tool",
    "conflict_no_upload_external",
    "conflict_no_monitoring_tool",
    "conflict_no_search_tool",
    "conflict_no_file_tool",
    "conflict_no_planning_tool",
    "conflict_no_control_external",
    "conflict_companion_proactive",
    "opportunity_agentic_control",
    "opportunity_recommendation",
]

CROSS_FEATURES = [
    "agentic_x_agent",
    "companion_x_intrusive",
    "boundary_x_external",
    "actionable_x_browser",
    "actionable_x_file",
    "actionable_x_agent",
    "refusal_x_monitoring",
    "refusal_x_message",
]

FEATURE_NAMES = CONV_FEATURES + TOOL_FEATURES + INTERACTION_FEATURES + CROSS_FEATURES
FEATURE_GROUPS = {
    "conv": CONV_FEATURES,
    "tool": TOOL_FEATURES,
    "interaction": INTERACTION_FEATURES,
    "cross": CROSS_FEATURES,
}

REFUSAL_PATTERNS = [
    r"\bdo not\b",
    r"\bdon't\b",
    r"\bno need\b",
    r"\bnot now\b",
    r"不要",
    r"别",
    r"不用",
    r"先别",
    r"不想",
    r"别帮",
]

JUST_VENTING_PATTERNS = [
    r"只是想",
    r"吐槽",
    r"随便看看",
    r"just looking",
    r"just vent",
    r"陪我",
    r"安静",
]

QUIET_PATTERNS = [
    r"安静",
    r"陪着",
    r"别打扰",
    r"放着",
    r"慢慢看",
    r"quiet",
]

DOMAIN_PATTERNS = {
    "file": [r"file", r"folder", r"pdf", r"doc", r"csv", r"文件", r"合同", r"本地"],
    "browser": [r"browser", r"web", r"url", r"网页", r"页面", r"链接", r"打开"],
    "search": [r"search", r"lookup", r"find", r"检索", r"搜索", r"查找"],
    "message": [r"sms", r"discord", r"wechat", r"email", r"reply", r"message", r"短信", r"回复"],
    "calendar": [r"calendar", r"schedule", r"日历", r"行程", r"安排"],
    "reminder": [r"remind", r"todo", r"task", r"habit", r"提醒", r"待办"],
    "monitoring": [r"monitor", r"listen", r"watch", r"camera", r"detect", r"监听", r"监控", r"观察"],
    "emotion": [r"emotion", r"mood", r"stress", r"lamp", r"情绪", r"心情", r"压力", r"灯"],
    "audio": [r"audio", r"sound", r"music", r"voice", r"声音", r"音乐"],
    "camera": [r"camera", r"photo", r"vision", r"摄像", r"照片"],
    "home": [r"home", r"clean", r"robot", r"air", r"plant", r"家", r"清洁", r"空气", r"植物"],
    "travel": [r"travel", r"trip", r"itinerary", r"ticket", r"hotel", r"旅行", r"行程", r"机票", r"酒店"],
    "code": [r"code", r"vscode", r"github", r"编程", r"代码"],
    "research": [r"paper", r"research", r"archive", r"论文", r"资料", r"引用"],
    "study": [r"study", r"anki", r"reading", r"学习", r"阅读", r"报名"],
    "social": [r"mastodon", r"xiaohongshu", r"post", r"social", r"社交", r"发布"],
}

EXTERNAL_PATTERNS = [
    r"send",
    r"post",
    r"reply",
    r"upload",
    r"api",
    r"connector",
    r"browser",
    r"web",
    r"发送",
    r"发布",
    r"上传",
]

PROACTIVE_PATTERNS = [
    r"proactive",
    r"remind",
    r"monitor",
    r"watch",
    r"listen",
    r"auto",
    r"提醒",
    r"监听",
    r"监控",
    r"自动",
]


def build_policy_features(config: dict) -> PolicyFeatureBuilder | None:
    cfg = config.get("policy_features", {})
    if not cfg.get("enabled", False):
        return None
    groups = tuple(cfg.get("groups", FEATURE_GROUPS.keys()))
    unknown = sorted(set(groups) - set(FEATURE_GROUPS))
    if unknown:
        raise ValueError(f"unknown policy feature groups: {unknown}")
    return PolicyFeatureBuilder(enabled=True, groups=groups)


def _conversation_features(example: PairExample) -> dict[str, float]:
    text = example.conversation_text.lower()
    scenario = example.scenario_type.lower()
    relevance = example.relevance_mode.lower()
    features = {
        "conv_agentic_task": _score_any(scenario, [r"agentic"]) or _score_any(text, [r"帮我", r"please", r"can you", r"打开", r"处理", r"calculate", r"fill"]),
        "conv_companion": _score_any(scenario + " " + relevance, [r"companion", r"no_tool"]) or _score_any(text, [r"陪", r"聊", r"吐槽"]),
        "conv_boundary": _score_any(scenario + " " + relevance, [r"boundary", r"refusal"]) or _score_any(text, REFUSAL_PATTERNS),
        "conv_proactive": _score_any(scenario + " " + relevance, [r"proactive", r"followup"]),
        "conv_open_thread": _score_any(scenario + " " + relevance, [r"open_thread"]),
        "conv_actionable": _score_any(text, [r"帮我", r"please", r"can you", r"处理", r"计算", r"打开", r"填", r"生成", r"整理"]),
        "conv_authorized": _score_any(text, [r"帮我", r"please", r"can you", r"直接", r"按我说的", r"我授权"]),
        "conv_refusal": _score_any(text, REFUSAL_PATTERNS),
        "conv_no_tool": _score_any(text, [r"不要.*工具", r"别.*帮", r"不用.*帮", r"不要.*变成", r"no tool", r"no need"]),
        "conv_no_reminder": _score_any(text, [r"别.*提醒", r"不要.*提醒", r"不用.*提醒", r"no reminder"]),
        "conv_no_upload": _score_any(text, [r"别.*上传", r"不要.*上传", r"don't upload", r"do not upload"]),
        "conv_no_monitoring": _score_any(text, [r"别.*监听", r"不要.*监听", r"不想.*监听", r"别.*监控", r"monitoring"]),
        "conv_no_search": _score_any(text, [r"别.*搜", r"不要.*搜", r"别.*查", r"不要.*查", r"don't search"]),
        "conv_no_file": _score_any(text, [r"别.*文件", r"不要.*文件", r"别.*索引", r"不要.*索引", r"file"]),
        "conv_no_planning": _score_any(text, [r"别.*规划", r"不要.*规划", r"别.*行程", r"不要.*行程", r"itinerary"]),
        "conv_no_control": _score_any(text, [r"别.*控制", r"不要.*控制", r"不要管", r"don't control"]),
        "conv_just_venting": _score_any(text, JUST_VENTING_PATTERNS),
        "conv_quiet_presence": _score_any(text, QUIET_PATTERNS),
    }
    return {name: float(value) for name, value in features.items()}


def _tool_features(example: PairExample) -> dict[str, float]:
    text = f"{example.tool_id}\n{example.tool_text}".lower()
    kind = "agent" if example.tool_id.startswith("agent.") else "plugin"
    features = {
        "tool_agent": 1.0 if kind == "agent" else 0.0,
        "tool_plugin": 1.0 if kind == "plugin" else 0.0,
        "tool_external": _score_any(text, EXTERNAL_PATTERNS),
        "tool_proactive": _score_any(text, PROACTIVE_PATTERNS),
    }
    for domain, patterns in DOMAIN_PATTERNS.items():
        features[f"tool_{domain}"] = _score_any(text, patterns)
    return {name: float(features.get(name, 0.0)) for name in TOOL_FEATURES}


def _assemble_features(
    example: PairExample,
    conv: dict[str, float],
    tool: dict[str, float],
    *,
    groups: tuple[str, ...],
) -> list[float]:
    del example
    domain_conflict = max(
        conv["conv_no_reminder"] * tool["tool_reminder"],
        conv["conv_no_upload"] * tool["tool_external"],
        conv["conv_no_monitoring"] * max(tool["tool_monitoring"], tool["tool_emotion"]),
        conv["conv_no_search"] * tool["tool_search"],
        conv["conv_no_file"] * tool["tool_file"],
        conv["conv_no_planning"] * max(tool["tool_travel"], tool["tool_calendar"]),
        conv["conv_no_control"] * tool["tool_external"],
    )
    interaction = {
        "conflict_refusal_domain": conv["conv_refusal"] * max(domain_conflict, tool["tool_proactive"]),
        "conflict_no_reminder_tool": conv["conv_no_reminder"] * tool["tool_reminder"],
        "conflict_no_upload_external": conv["conv_no_upload"] * tool["tool_external"],
        "conflict_no_monitoring_tool": conv["conv_no_monitoring"] * max(tool["tool_monitoring"], tool["tool_emotion"]),
        "conflict_no_search_tool": conv["conv_no_search"] * tool["tool_search"],
        "conflict_no_file_tool": conv["conv_no_file"] * tool["tool_file"],
        "conflict_no_planning_tool": conv["conv_no_planning"] * max(tool["tool_travel"], tool["tool_calendar"]),
        "conflict_no_control_external": conv["conv_no_control"] * tool["tool_external"],
        "conflict_companion_proactive": conv["conv_companion"] * tool["tool_proactive"],
        "opportunity_agentic_control": conv["conv_agentic_task"] * max(tool["tool_agent"], tool["tool_browser"], tool["tool_file"]),
        "opportunity_recommendation": max(conv["conv_proactive"], conv["conv_open_thread"]) * (1.0 - conv["conv_refusal"]),
    }
    cross = {
        "agentic_x_agent": conv["conv_agentic_task"] * tool["tool_agent"],
        "companion_x_intrusive": conv["conv_companion"] * max(tool["tool_proactive"], tool["tool_monitoring"], tool["tool_external"]),
        "boundary_x_external": conv["conv_boundary"] * tool["tool_external"],
        "actionable_x_browser": conv["conv_actionable"] * tool["tool_browser"],
        "actionable_x_file": conv["conv_actionable"] * tool["tool_file"],
        "actionable_x_agent": conv["conv_actionable"] * tool["tool_agent"],
        "refusal_x_monitoring": conv["conv_refusal"] * max(tool["tool_monitoring"], tool["tool_emotion"]),
        "refusal_x_message": conv["conv_refusal"] * tool["tool_message"],
    }
    merged = {**conv, **tool, **interaction, **cross}
    return [float(np.clip(merged[name], -1.0, 1.0)) for name in _feature_names_for_groups(groups)]


def _feature_names_for_groups(groups: tuple[str, ...]) -> list[str]:
    names = []
    for group in groups:
        names.extend(FEATURE_GROUPS[group])
    return names


def _score_any(text: str, patterns: list[str]) -> float:
    score = 0.0
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            score += 1.0
    return float(min(1.0, score / max(1.0, min(3.0, len(patterns) / 2))))
