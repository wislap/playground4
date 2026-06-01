from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ToolKind = Literal["agent", "plugin"]
MessageRole = Literal["system", "user", "assistant"]
CategorySource = Literal["manual", "heuristic", "llm", "mixed"]
ScoreAxis = Literal[
    "capability_match",
    "action_demand",
    "target_specificity",
    "consent_boundary",
    "intervention_cost",
    "companionship_fit",
    "final_preference",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolRecord(StrictModel):
    tool_id: str
    kind: ToolKind
    source_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tool_id", "source_text")
    @classmethod
    def require_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value


class ToolUniverse(StrictModel):
    tool_universe_id: str
    created_at: str | None = None
    tools: list[ToolRecord]

    @model_validator(mode="after")
    def validate_unique_tool_ids(self) -> ToolUniverse:
        tool_ids = [tool.tool_id for tool in self.tools]
        if len(tool_ids) != len(set(tool_ids)):
            raise ValueError("tool ids must be unique")
        return self

    @property
    def agents(self) -> list[ToolRecord]:
        return [tool for tool in self.tools if tool.kind == "agent"]

    @property
    def plugins(self) -> list[ToolRecord]:
        return [tool for tool in self.tools if tool.kind == "plugin"]

    def by_id(self) -> dict[str, ToolRecord]:
        return {tool.tool_id: tool for tool in self.tools}


class ToolUniverseManifest(StrictModel):
    tool_universe_id: str
    schema_version: str = "tool_universe_schema_v1"
    created_at: str | None = None
    kind: str = "real"
    tool_count: int
    source: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tool_count")
    @classmethod
    def validate_tool_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("tool_count must be non-negative")
        return value


class ToolCategory(StrictModel):
    tool_id: str
    kind: ToolKind
    primary_category: str
    secondary_categories: list[str] = Field(default_factory=list)
    generation_tags: list[str] = Field(default_factory=list)
    source: CategorySource = "manual"

    @field_validator("primary_category")
    @classmethod
    def require_primary_category(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("primary_category must be non-empty")
        return value

class ToolClassification(StrictModel):
    tool_universe_id: str
    taxonomy_version: str
    categories: list[ToolCategory]

    @model_validator(mode="after")
    def validate_unique_tool_ids(self) -> ToolClassification:
        tool_ids = [category.tool_id for category in self.categories]
        if len(tool_ids) != len(set(tool_ids)):
            raise ValueError("classification tool ids must be unique")
        return self

    def by_id(self) -> dict[str, ToolCategory]:
        return {category.tool_id: category for category in self.categories}


class Message(StrictModel):
    role: MessageRole
    text: str = ""
    attachments: list[str] = Field(default_factory=list)
    timestamp: float | None = None


class GenerationHint(StrictModel):
    target_tool_ids: list[str] = Field(default_factory=list)
    scenario_id: str | None = None
    language: str | None = None


class Provenance(StrictModel):
    model: str | None = None
    prompt_version: str | None = None
    seed: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ConversationRecord(StrictModel):
    conversation_id: str
    messages: list[Message]
    trigger: str = "turn_end"
    generation_hint: GenerationHint = Field(default_factory=GenerationHint)
    provenance: Provenance = Field(default_factory=Provenance)

    @model_validator(mode="after")
    def validate_messages(self) -> ConversationRecord:
        if not self.messages:
            raise ValueError("conversation must contain messages")
        return self


class CandidateSet(StrictModel):
    sampling_seed: int
    plugin_sample_rate: float
    tool_ids: list[str]

    @field_validator("plugin_sample_rate")
    @classmethod
    def validate_sample_rate(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("plugin_sample_rate must be in [0, 1]")
        return value

    @model_validator(mode="after")
    def validate_unique_tool_ids(self) -> CandidateSet:
        if len(self.tool_ids) != len(set(self.tool_ids)):
            raise ValueError("candidate tool ids must be unique")
        return self


class CandidateSetRecord(StrictModel):
    sample_id: str
    conversation_id: str
    candidate_set: CandidateSet


class RawToolScore(StrictModel):
    tool_id: str
    raw_score: float

    @field_validator("raw_score")
    @classmethod
    def validate_raw_score(cls, value: float) -> float:
        if not 0.0 <= value <= 100.0:
            raise ValueError("raw_score must be in [0, 100]")
        return value


class RawAxisScore(StrictModel):
    axis: ScoreAxis
    raw_score: float

    @field_validator("raw_score")
    @classmethod
    def validate_raw_score(cls, value: float) -> float:
        if not 0.0 <= value <= 100.0:
            raise ValueError("raw_score must be in [0, 100]")
        return value


class MultiAxisToolScore(StrictModel):
    tool_id: str
    scores: list[RawAxisScore]
    rationale: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_unique_axes(self) -> MultiAxisToolScore:
        axes = [score.axis for score in self.scores]
        if len(axes) != len(set(axes)):
            raise ValueError("axis scores must not contain duplicate axes")
        return self


class JudgmentRecord(StrictModel):
    sample_id: str
    conversation_id: str
    judge_run_id: str
    candidate_tool_order: list[str]
    scores: list[RawToolScore]
    provenance: Provenance = Field(default_factory=Provenance)

    @model_validator(mode="after")
    def validate_scores_match_order(self) -> JudgmentRecord:
        ordered = self.candidate_tool_order
        scored = [score.tool_id for score in self.scores]
        if set(ordered) != set(scored):
            raise ValueError("scores must cover the candidate tool order exactly")
        if len(scored) != len(set(scored)):
            raise ValueError("scores must not contain duplicate tool ids")
        return self


class MultiAxisJudgmentRecord(StrictModel):
    sample_id: str
    conversation_id: str
    judge_run_id: str
    candidate_tool_order: list[str]
    scores: list[MultiAxisToolScore]
    provenance: Provenance = Field(default_factory=Provenance)

    @model_validator(mode="after")
    def validate_scores_match_order(self) -> MultiAxisJudgmentRecord:
        ordered = self.candidate_tool_order
        scored = [score.tool_id for score in self.scores]
        if set(ordered) != set(scored):
            raise ValueError("scores must cover the candidate tool order exactly")
        if len(scored) != len(set(scored)):
            raise ValueError("scores must not contain duplicate tool ids")
        return self


class TargetScore(StrictModel):
    tool_id: str
    confidence: float


class MultiAxisTargetScore(StrictModel):
    tool_id: str
    scores: dict[ScoreAxis, float]
    raw_scores: dict[ScoreAxis, float] = Field(default_factory=dict)
    local_z: dict[ScoreAxis, float] = Field(default_factory=dict)
    judge_disagreement: dict[ScoreAxis, float] = Field(default_factory=dict)


class SampleQuality(StrictModel):
    top_raw_score: float | None = None
    judge_disagreement: float | None = None
    has_high_confidence_candidate: bool | None = None


class DatasetProvenance(StrictModel):
    conversation_model: str | None = None
    judge_model: str | None = None
    generator_prompt_version: str | None = None
    judge_prompt_version: str | None = None
    calibration_version: str


class DatasetSample(StrictModel):
    sample_id: str
    split: Literal["train", "valid", "test"]
    tool_universe_id: str
    conversation: dict[str, Any]
    candidate_set: CandidateSet
    targets: list[TargetScore]
    quality: SampleQuality = Field(default_factory=SampleQuality)
    provenance: DatasetProvenance

    @model_validator(mode="after")
    def validate_targets_match_candidates(self) -> DatasetSample:
        candidates = set(self.candidate_set.tool_ids)
        targets = [target.tool_id for target in self.targets]
        if set(targets) != candidates:
            raise ValueError("targets must cover candidate_set.tool_ids exactly")
        if len(targets) != len(set(targets)):
            raise ValueError("targets must not contain duplicate tool ids")
        return self
