from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from v22b_dataset_loader import PairExample


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

ACTION_CONSENT_FEATURES = [
    "ac_no_call_intent",
    "ac_ask_confirm_intent",
    "ac_can_call_intent",
    "ac_explicit_refusal",
    "ac_temporary_defer",
    "ac_just_venting",
    "ac_quiet_presence",
    "ac_explicit_authorization",
    "ac_denies_search",
    "ac_denies_monitoring",
    "ac_denies_upload",
    "ac_denies_message",
    "ac_denies_control",
    "ac_denies_file_access",
    "ac_requires_final_confirmation",
    "ac_requested_search",
    "ac_requested_browser",
    "ac_requested_form_fill",
    "ac_requested_save_template",
    "ac_requested_message_reply",
    "ac_requested_file_access",
    "ac_requested_monitoring",
    "ac_requested_device_control",
    "ac_tool_read_only",
    "ac_tool_external_side_effect",
    "ac_tool_continuous_monitoring",
    "ac_tool_device_control",
    "ac_tool_agentic_control",
    "ac_tool_final_submit_risk",
    "ac_match_requested_search",
    "ac_match_requested_browser",
    "ac_match_requested_form_fill",
    "ac_match_requested_save_template",
    "ac_match_requested_message_reply",
    "ac_match_requested_file_access",
    "ac_match_requested_monitoring",
    "ac_match_requested_device_control",
    "ac_conflict_denied_search",
    "ac_conflict_denied_monitoring",
    "ac_conflict_denied_upload",
    "ac_conflict_denied_message",
    "ac_conflict_denied_control",
    "ac_conflict_denied_file_access",
    "ac_conflict_final_submit",
    "ac_no_call_side_effect_risk",
]

ACTION_CONSENT_B_FEATURES = [
    "acb_no_call_intent",
    "acb_ask_confirm_intent",
    "acb_can_call_intent",
    "acb_temporary_defer",
    "acb_final_confirmation_required",
    "acb_tool_read_only",
    "acb_tool_local_write",
    "acb_tool_external_action",
    "acb_tool_background_monitor",
    "acb_tool_user_state_inference",
    "acb_tool_device_or_environment_control",
    "acb_tool_agentic_operation",
    "acb_tool_commit_or_submit",
    "acb_weak_operation_match",
    "acb_strong_operation_match",
    "acb_denied_operation_conflict",
    "acb_no_call_side_effect_conflict",
    "acb_confirmation_boundary_conflict",
    "acb_refusal_background_monitor_veto",
    "acb_refusal_user_state_veto",
    "acb_refusal_external_action_veto",
    "acb_refusal_device_control_veto",
]

FEATURE_NAMES = (
    CONV_FEATURES
    + TOOL_FEATURES
    + INTERACTION_FEATURES
    + CROSS_FEATURES
    + ACTION_CONSENT_FEATURES
    + ACTION_CONSENT_B_FEATURES
)
FEATURE_GROUPS = {
    "conv": CONV_FEATURES,
    "tool": TOOL_FEATURES,
    "interaction": INTERACTION_FEATURES,
    "cross": CROSS_FEATURES,
    "action_consent": ACTION_CONSENT_FEATURES,
    "action_consent_b": ACTION_CONSENT_B_FEATURES,
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

TEMPORARY_DEFER_PATTERNS = [
    r"先别",
    r"现在.*别",
    r"暂时.*别",
    r"等.*再",
    r"not now",
    r"later",
    r"don't.*now",
]

CONFIRMATION_PATTERNS = [
    r"提交前.*确认",
    r"发送前.*确认",
    r"发布前.*确认",
    r"先.*确认",
    r"别.*提交",
    r"不要.*提交",
    r"before.*submit",
    r"before.*send",
    r"confirm.*before",
]

REQUEST_PATTERNS = {
    "search": [r"搜", r"搜索", r"查", r"检索", r"找.*资料", r"lookup", r"search", r"find"],
    "browser": [r"打开.*链接", r"打开.*网页", r"网页", r"url", r"链接", r"browser", r"web"],
    "form_fill": [r"填.*表", r"填写", r"报名", r"表单", r"form", r"apply"],
    "save_template": [r"存.*模板", r"保存.*模板", r"话术", r"文本片段", r"snippet", r"template"],
    "message_reply": [r"回复", r"私信", r"短信", r"邮件", r"send", r"reply", r"message", r"email"],
    "file_access": [r"文件", r"资料", r"pdf", r"文档", r"folder", r"file", r"doc"],
    "monitoring": [r"监听", r"监控", r"提醒", r"watch", r"listen", r"monitor", r"detect"],
    "device_control": [r"调灯", r"开灯", r"关灯", r"控制", r"香薰", r"设备", r"灯光", r"control"],
}

OPERATION_PATTERNS = {
    "read_or_fetch": [
        r"读",
        r"读取",
        r"查看",
        r"打开",
        r"检索",
        r"搜索",
        r"查找",
        r"retrieve",
        r"read",
        r"fetch",
        r"search",
        r"lookup",
    ],
    "local_write": [
        r"保存",
        r"记录",
        r"写入",
        r"创建",
        r"整理",
        r"模板",
        r"片段",
        r"save",
        r"write",
        r"create",
        r"template",
        r"snippet",
    ],
    "external_action": [
        r"发送",
        r"发布",
        r"上传",
        r"回复",
        r"私信",
        r"邮件",
        r"api",
        r"send",
        r"post",
        r"upload",
        r"reply",
        r"message",
        r"email",
    ],
    "agentic_operation": [
        r"帮我",
        r"打开.*网页",
        r"填写",
        r"填.*表",
        r"处理",
        r"执行",
        r"自动",
        r"browser",
        r"web",
        r"form",
        r"execute",
        r"automate",
    ],
    "background_monitor": [
        r"监听",
        r"监控",
        r"后台",
        r"提醒",
        r"自动回复",
        r"monitor",
        r"watch",
        r"listen",
        r"detect",
        r"auto.?reply",
    ],
    "user_state_inference": [
        r"情绪",
        r"心情",
        r"语气",
        r"压力",
        r"状态",
        r"emotion",
        r"mood",
        r"tone",
        r"stress",
    ],
    "device_or_environment_control": [
        r"调灯",
        r"开灯",
        r"关灯",
        r"灯光",
        r"香薰",
        r"设备",
        r"清洁",
        r"机器人",
        r"device",
        r"light",
        r"lamp",
        r"diffuser",
        r"robot",
        r"control",
    ],
    "commit_or_submit": [
        r"提交",
        r"确认提交",
        r"发送",
        r"发布",
        r"下单",
        r"submit",
        r"commit",
        r"send",
        r"post",
        r"purchase",
    ],
}

DENIAL_PATTERNS = {
    "search": [r"别.*搜", r"不要.*搜", r"别.*查", r"不要.*查", r"先别.*查", r"don't search", r"do not search"],
    "monitoring": [r"别.*监听", r"不要.*监听", r"不想.*监听", r"别.*监控", r"不要.*监控", r"别.*看", r"don't monitor"],
    "upload": [r"别.*上传", r"不要.*上传", r"don't upload", r"do not upload"],
    "message": [r"别.*回复", r"不要.*回复", r"别.*发", r"不要.*发", r"don't send", r"do not send"],
    "control": [r"别.*控制", r"不要.*控制", r"不要管", r"别.*调灯", r"不要.*调灯", r"don't control"],
    "file_access": [r"别.*资料", r"不要.*资料", r"别.*文件", r"不要.*文件", r"别.*索引", r"不要.*索引"],
}

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


def build_policy_features(config: dict, *, section: str = "policy_features") -> PolicyFeatureBuilder | None:
    cfg = config.get(section, {})
    if not cfg.get("enabled", False):
        return None
    groups = tuple(cfg.get("groups", FEATURE_GROUPS.keys()))
    unknown = sorted(set(groups) - set(FEATURE_GROUPS))
    if unknown:
        raise ValueError(f"unknown policy feature groups: {unknown}")
    return PolicyFeatureBuilder(enabled=True, groups=groups)


def _conversation_features(example: PairExample) -> dict[str, float]:
    text = example.conversation_text.lower()
    latest_user = _latest_user_or_conversation(example)
    scenario = example.scenario_type.lower()
    relevance = example.relevance_mode.lower()
    actionability = example.latest_user_actionability.lower()
    authorization = example.authorization_level.lower()
    features = {
        "conv_agentic_task": _score_any(scenario, [r"agentic"]) or _score_any(text, [r"帮我", r"please", r"can you", r"打开", r"处理", r"calculate", r"fill"]),
        "conv_companion": _score_any(scenario + " " + relevance, [r"companion", r"no_tool"]) or _score_any(text, [r"陪", r"聊", r"吐槽"]),
        "conv_boundary": _score_any(scenario + " " + relevance, [r"boundary", r"refusal"]) or _score_any(text, REFUSAL_PATTERNS),
        "conv_proactive": _score_any(scenario + " " + relevance, [r"proactive", r"followup"]),
        "conv_open_thread": _score_any(scenario + " " + relevance, [r"open_thread"]),
        "conv_actionable": (1.0 if actionability == "actionable" else 0.0)
        or _score_any(latest_user, [r"帮我", r"please", r"can you", r"处理", r"计算", r"打开", r"填", r"生成", r"整理", r"要", r"yes"]),
        "conv_authorized": (1.0 if authorization in {"explicit", "implied"} and actionability == "actionable" else 0.0)
        or _score_any(latest_user, [r"帮我", r"please", r"can you", r"直接", r"按我说的", r"我授权", r"actually yes", r"\byes\b", r"要"]),
        "conv_refusal": _score_any(latest_user, REFUSAL_PATTERNS),
        "conv_no_tool": _score_any(latest_user, [r"不要.*工具", r"别.*帮", r"不用.*帮", r"不要.*变成", r"no tool", r"no need", r"just chat"]),
        "conv_no_reminder": _score_any(text, [r"别.*提醒", r"不要.*提醒", r"不用.*提醒", r"no reminder"]),
        "conv_no_upload": _score_any(text, [r"别.*上传", r"不要.*上传", r"don't upload", r"do not upload"]),
        "conv_no_monitoring": _score_any(latest_user, [r"别.*监听", r"不要.*监听", r"不想.*监听", r"别.*监控", r"monitoring"]),
        "conv_no_search": _score_any(latest_user, [r"别.*搜", r"不要.*搜", r"别.*查", r"不要.*查", r"不查", r"don't search"]),
        "conv_no_file": _score_any(text, [r"别.*文件", r"不要.*文件", r"别.*索引", r"不要.*索引", r"file"]),
        "conv_no_planning": _score_any(text, [r"别.*规划", r"不要.*规划", r"别.*行程", r"不要.*行程", r"itinerary"]),
        "conv_no_control": _score_any(latest_user, [r"别.*控制", r"不要.*控制", r"不要管", r"don't control"]),
        "conv_just_venting": _score_any(latest_user, JUST_VENTING_PATTERNS),
        "conv_quiet_presence": _score_any(latest_user, QUIET_PATTERNS),
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
    action_consent = _action_consent_features(example, conv, tool)
    action_consent_b = _action_consent_b_features(example, conv, tool)
    merged = {**conv, **tool, **interaction, **cross, **action_consent, **action_consent_b}
    return [float(np.clip(merged[name], -1.0, 1.0)) for name in _feature_names_for_groups(groups)]


def _action_consent_features(
    example: PairExample,
    conv: dict[str, float],
    tool: dict[str, float],
) -> dict[str, float]:
    text = example.conversation_text.lower()
    latest_user = _latest_user_or_conversation(example)
    scenario = example.scenario_type.lower()
    relevance = example.relevance_mode.lower()
    actionability = example.latest_user_actionability.lower()
    authorization = example.authorization_level.lower()
    tool_text = f"{example.tool_id}\n{example.tool_text}".lower()

    no_call_intent = max(
        conv["conv_no_tool"],
        conv["conv_just_venting"],
        conv["conv_quiet_presence"],
        _score_any(scenario + " " + relevance, [r"no_tool", r"companion", r"emotional_support", r"memory_recall"]),
    )
    ask_confirm_intent = max(
        _score_any(scenario + " " + relevance, [r"weak", r"clarification", r"ambiguous", r"suggested_tool_no_auth"]),
        _score_any(latest_user, [r"要不要", r"可以吗", r"先问", r"确认", r"maybe", r"should i"]),
    )
    can_call_intent = max(
        conv["conv_authorized"],
        1.0 if actionability == "actionable" and authorization in {"explicit", "implied"} else 0.0,
        _score_any(scenario + " " + relevance, [r"agentic_task", r"explicit_plugin_action", r"actionable_tool"]),
    )
    explicit_refusal = max(conv["conv_refusal"], _score_any(text, REFUSAL_PATTERNS))
    temporary_defer = _score_any(latest_user, TEMPORARY_DEFER_PATTERNS)

    denied = {name: _score_any(latest_user, patterns) for name, patterns in DENIAL_PATTERNS.items()}
    requested = {name: _score_any(latest_user, patterns) for name, patterns in REQUEST_PATTERNS.items()}

    tool_read_only = max(tool["tool_file"], tool["tool_research"], tool["tool_study"]) * (1.0 - tool["tool_external"])
    tool_external_side_effect = max(tool["tool_external"], tool["tool_message"], tool["tool_social"])
    tool_continuous_monitoring = max(tool["tool_monitoring"], tool["tool_camera"], tool["tool_emotion"])
    tool_device_control = max(tool["tool_home"], tool["tool_audio"], tool["tool_emotion"])
    tool_agentic_control = max(tool["tool_agent"], tool["tool_browser"])
    tool_final_submit_risk = max(tool_external_side_effect, tool_agentic_control)

    tool_action = {
        "search": max(tool["tool_search"], tool["tool_research"]),
        "browser": tool["tool_browser"],
        "form_fill": max(tool["tool_browser"], _score_any(tool_text, [r"form", r"表单", r"报名", r"apply"])),
        "save_template": _score_any(tool_text, [r"snippet", r"template", r"模板", r"话术", r"文本片段"]),
        "message_reply": tool["tool_message"],
        "file_access": tool["tool_file"],
        "monitoring": tool_continuous_monitoring,
        "device_control": tool_device_control,
    }

    match = {name: requested[name] * tool_action[name] for name in requested}
    conflict = {
        "search": denied["search"] * tool_action["search"],
        "monitoring": denied["monitoring"] * tool_action["monitoring"],
        "upload": denied["upload"] * tool_external_side_effect,
        "message": denied["message"] * tool_action["message_reply"],
        "control": denied["control"] * max(tool_action["device_control"], tool_agentic_control),
        "file_access": denied["file_access"] * tool_action["file_access"],
    }
    conflict_final_submit = _score_any(text, CONFIRMATION_PATTERNS) * tool_final_submit_risk
    side_effect_risk = max(tool_external_side_effect, tool_continuous_monitoring, tool_device_control, tool_agentic_control)

    features = {
        "ac_no_call_intent": no_call_intent,
        "ac_ask_confirm_intent": ask_confirm_intent,
        "ac_can_call_intent": can_call_intent,
        "ac_explicit_refusal": explicit_refusal,
        "ac_temporary_defer": temporary_defer,
        "ac_just_venting": conv["conv_just_venting"],
        "ac_quiet_presence": conv["conv_quiet_presence"],
        "ac_explicit_authorization": conv["conv_authorized"],
        "ac_denies_search": denied["search"],
        "ac_denies_monitoring": denied["monitoring"],
        "ac_denies_upload": denied["upload"],
        "ac_denies_message": denied["message"],
        "ac_denies_control": denied["control"],
        "ac_denies_file_access": denied["file_access"],
        "ac_requires_final_confirmation": _score_any(text, CONFIRMATION_PATTERNS),
        "ac_requested_search": requested["search"],
        "ac_requested_browser": requested["browser"],
        "ac_requested_form_fill": requested["form_fill"],
        "ac_requested_save_template": requested["save_template"],
        "ac_requested_message_reply": requested["message_reply"],
        "ac_requested_file_access": requested["file_access"],
        "ac_requested_monitoring": requested["monitoring"],
        "ac_requested_device_control": requested["device_control"],
        "ac_tool_read_only": tool_read_only,
        "ac_tool_external_side_effect": tool_external_side_effect,
        "ac_tool_continuous_monitoring": tool_continuous_monitoring,
        "ac_tool_device_control": tool_device_control,
        "ac_tool_agentic_control": tool_agentic_control,
        "ac_tool_final_submit_risk": tool_final_submit_risk,
        "ac_match_requested_search": match["search"],
        "ac_match_requested_browser": match["browser"],
        "ac_match_requested_form_fill": match["form_fill"],
        "ac_match_requested_save_template": match["save_template"],
        "ac_match_requested_message_reply": match["message_reply"],
        "ac_match_requested_file_access": match["file_access"],
        "ac_match_requested_monitoring": match["monitoring"],
        "ac_match_requested_device_control": match["device_control"],
        "ac_conflict_denied_search": conflict["search"],
        "ac_conflict_denied_monitoring": conflict["monitoring"],
        "ac_conflict_denied_upload": conflict["upload"],
        "ac_conflict_denied_message": conflict["message"],
        "ac_conflict_denied_control": conflict["control"],
        "ac_conflict_denied_file_access": conflict["file_access"],
        "ac_conflict_final_submit": conflict_final_submit,
        "ac_no_call_side_effect_risk": no_call_intent * side_effect_risk,
    }
    return {name: float(features[name]) for name in ACTION_CONSENT_FEATURES}


def _action_consent_b_features(
    example: PairExample,
    conv: dict[str, float],
    tool: dict[str, float],
) -> dict[str, float]:
    text = example.conversation_text.lower()
    latest_user = _latest_user_or_conversation(example)
    scenario = example.scenario_type.lower()
    relevance = example.relevance_mode.lower()
    actionability = example.latest_user_actionability.lower()
    authorization = example.authorization_level.lower()
    tool_text = f"{example.tool_id}\n{example.tool_text}".lower()

    no_call_intent = max(
        conv["conv_no_tool"],
        conv["conv_just_venting"],
        conv["conv_quiet_presence"],
        _score_any(scenario + " " + relevance, [r"no_tool", r"companion", r"emotional_support", r"memory_recall"]),
    )
    ask_confirm_intent = max(
        _score_any(scenario + " " + relevance, [r"weak", r"clarification", r"ambiguous", r"suggested_tool_no_auth"]),
        _score_any(latest_user, [r"要不要", r"可以吗", r"先问", r"确认", r"maybe", r"should i"]),
    )
    can_call_intent = max(
        conv["conv_authorized"],
        1.0 if actionability == "actionable" and authorization in {"explicit", "implied"} else 0.0,
        _score_any(scenario + " " + relevance, [r"agentic_task", r"explicit_plugin_action", r"actionable_tool"]),
    )
    temporary_defer = _score_any(latest_user, TEMPORARY_DEFER_PATTERNS)
    final_confirmation_required = _score_any(latest_user, CONFIRMATION_PATTERNS)

    requested = {name: _score_any(latest_user, patterns) for name, patterns in OPERATION_PATTERNS.items()}
    denied = {name: _score_any(latest_user, patterns) for name, patterns in DENIAL_PATTERNS.items()}

    tool_read_or_fetch = max(
        _score_any(tool_text, OPERATION_PATTERNS["read_or_fetch"]),
        tool["tool_file"],
        tool["tool_search"],
        tool["tool_research"],
        tool["tool_browser"],
    )
    tool_local_write = _score_any(tool_text, OPERATION_PATTERNS["local_write"])
    tool_external_action = max(
        _score_any(tool_text, OPERATION_PATTERNS["external_action"]),
        tool["tool_external"],
        tool["tool_message"],
        tool["tool_social"],
    )
    tool_background_monitor = max(
        _score_any(tool_text, OPERATION_PATTERNS["background_monitor"]),
        tool["tool_monitoring"],
        tool["tool_camera"],
        tool["tool_proactive"],
    )
    tool_user_state_inference = max(
        _score_any(tool_text, OPERATION_PATTERNS["user_state_inference"]),
        tool["tool_emotion"],
    )
    tool_device_or_environment_control = max(
        _score_any(tool_text, OPERATION_PATTERNS["device_or_environment_control"]),
        tool["tool_home"],
        tool["tool_audio"],
    )
    tool_agentic_operation = max(
        _score_any(tool_text, OPERATION_PATTERNS["agentic_operation"]),
        tool["tool_agent"],
        tool["tool_browser"],
    )
    tool_commit_or_submit = _score_any(tool_text, OPERATION_PATTERNS["commit_or_submit"])

    tool_side_effect = max(
        tool_external_action,
        tool_background_monitor,
        tool_user_state_inference,
        tool_device_or_environment_control,
        tool_agentic_operation,
        tool_commit_or_submit,
    )
    tool_read_only = tool_read_or_fetch * (1.0 - tool_side_effect)

    read_match = requested["read_or_fetch"] * tool_read_or_fetch
    local_write_match = requested["local_write"] * tool_local_write
    external_match = requested["external_action"] * tool_external_action
    agentic_match = requested["agentic_operation"] * tool_agentic_operation
    weak_operation_match = max(read_match, local_write_match, external_match, agentic_match)

    # Strong matches are limited to explicit action states. Background monitoring
    # and device/environment control intentionally do not create positive match
    # features here; they are too easy to over-lift in companionship scenes.
    strong_operation_match = can_call_intent * max(local_write_match, external_match, agentic_match)

    denied_operation_conflict = max(
        denied["search"] * tool_read_or_fetch,
        denied["upload"] * tool_external_action,
        denied["message"] * tool_external_action,
        denied["file_access"] * tool_read_or_fetch,
        denied["monitoring"] * max(tool_background_monitor, tool_user_state_inference),
        denied["control"] * max(tool_device_or_environment_control, tool_agentic_operation),
    )
    no_call_side_effect_conflict = no_call_intent * tool_side_effect
    confirmation_boundary_conflict = final_confirmation_required * max(tool_commit_or_submit, tool_external_action, tool_agentic_operation)

    refusal_background_monitor_veto = conv["conv_refusal"] * tool_background_monitor
    refusal_user_state_veto = conv["conv_refusal"] * tool_user_state_inference
    refusal_external_action_veto = conv["conv_refusal"] * tool_external_action
    refusal_device_control_veto = conv["conv_refusal"] * tool_device_or_environment_control

    features = {
        "acb_no_call_intent": no_call_intent,
        "acb_ask_confirm_intent": ask_confirm_intent,
        "acb_can_call_intent": can_call_intent,
        "acb_temporary_defer": temporary_defer,
        "acb_final_confirmation_required": final_confirmation_required,
        "acb_tool_read_only": tool_read_only,
        "acb_tool_local_write": tool_local_write,
        "acb_tool_external_action": tool_external_action,
        "acb_tool_background_monitor": tool_background_monitor,
        "acb_tool_user_state_inference": tool_user_state_inference,
        "acb_tool_device_or_environment_control": tool_device_or_environment_control,
        "acb_tool_agentic_operation": tool_agentic_operation,
        "acb_tool_commit_or_submit": tool_commit_or_submit,
        "acb_weak_operation_match": weak_operation_match,
        "acb_strong_operation_match": strong_operation_match,
        "acb_denied_operation_conflict": denied_operation_conflict,
        "acb_no_call_side_effect_conflict": no_call_side_effect_conflict,
        "acb_confirmation_boundary_conflict": confirmation_boundary_conflict,
        "acb_refusal_background_monitor_veto": refusal_background_monitor_veto,
        "acb_refusal_user_state_veto": refusal_user_state_veto,
        "acb_refusal_external_action_veto": refusal_external_action_veto,
        "acb_refusal_device_control_veto": refusal_device_control_veto,
    }
    return {name: float(features[name]) for name in ACTION_CONSENT_B_FEATURES}


def _latest_user_or_conversation(example: PairExample) -> str:
    latest_user = example.latest_user_text.strip().lower()
    if latest_user:
        return latest_user
    return example.conversation_text.lower()


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
