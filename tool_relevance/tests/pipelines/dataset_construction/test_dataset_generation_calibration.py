from statistics import fmean, pstdev

from tool_relevance_lab.dataset_generation.calibration import calibrate_confidences
from tool_relevance_lab.dataset_generation.schemas import JudgmentRecord, RawToolScore


def judgment(sample_id: str, scores: dict[str, float], run: int = 1) -> JudgmentRecord:
    return JudgmentRecord(
        sample_id=sample_id,
        conversation_id=sample_id.replace("sample", "conv"),
        judge_run_id=f"{sample_id}_run_{run}",
        candidate_tool_order=list(scores),
        scores=[RawToolScore(tool_id=tool_id, raw_score=score) for tool_id, score in scores.items()],
    )


def test_calibration_preserves_relative_high_scores() -> None:
    calibrated = calibrate_confidences(
        [
            judgment("sample_1", {"agent.browser_use": 90, "plugin.web_search": 60, "plugin.pdf": 10}),
            judgment("sample_2", {"agent.browser_use": 20, "plugin.web_search": 85, "plugin.pdf": 30}),
        ],
        clip_percentile=0.001,
    )

    by_key = {(row.sample_id, row.tool_id): row for row in calibrated}

    assert by_key[("sample_1", "agent.browser_use")].confidence > by_key[
        ("sample_1", "plugin.pdf")
    ].confidence
    assert by_key[("sample_2", "plugin.web_search")].confidence > by_key[
        ("sample_2", "agent.browser_use")
    ].confidence


def test_calibration_outputs_roughly_standardized_distribution() -> None:
    calibrated = calibrate_confidences(
        [
            judgment("sample_1", {"a": 90, "b": 60, "c": 10}),
            judgment("sample_2", {"a": 20, "b": 85, "c": 30}),
            judgment("sample_3", {"a": 10, "b": 20, "c": 95}),
            judgment("sample_4", {"a": 80, "b": 40, "c": 30}),
        ],
        clip_percentile=0.001,
    )
    confidences = [row.confidence for row in calibrated]

    assert abs(fmean(confidences)) < 0.1
    assert 0.7 < pstdev(confidences) < 1.2


def test_calibration_aggregates_repeated_judge_runs_by_median() -> None:
    calibrated = calibrate_confidences(
        [
            judgment("sample_1", {"a": 90, "b": 10}, run=1),
            judgment("sample_1", {"a": 80, "b": 20}, run=2),
        ],
        clip_percentile=0.001,
    )

    by_tool = {row.tool_id: row for row in calibrated}
    assert by_tool["a"].raw_score == 85
    assert by_tool["a"].judge_disagreement == 10
