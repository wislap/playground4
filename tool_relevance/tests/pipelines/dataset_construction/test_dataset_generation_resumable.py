import pytest

from tool_relevance_lab.dataset_generation.jsonl import append_jsonl, read_jsonl
from tool_relevance_lab.dataset_generation.resumable import run_resumable_jobs


@pytest.mark.anyio
async def test_run_resumable_jobs_skips_completed_items(tmp_path) -> None:
    output_path = tmp_path / "out.jsonl"
    error_path = tmp_path / "errors.jsonl"
    append_jsonl(output_path, [{"sample_id": "sample_1", "value": 1}])
    seen: list[str] = []

    async def worker(item: str) -> dict:
        seen.append(item)
        return {"sample_id": item, "value": 2}

    summary = await run_resumable_jobs(
        items=["sample_1", "sample_2"],
        item_id=lambda item: item,
        completed_id=lambda row: row.get("sample_id"),
        output_path=output_path,
        error_path=error_path,
        worker=worker,
        concurrency=2,
        progress_label="test",
    )

    rows = list(read_jsonl(output_path))
    assert seen == ["sample_2"]
    assert [row["sample_id"] for row in rows] == ["sample_1", "sample_2"]
    assert summary.completed_existing == 1
    assert summary.succeeded == 1
    assert summary.failed == 0


@pytest.mark.anyio
async def test_run_resumable_jobs_records_failures(tmp_path) -> None:
    output_path = tmp_path / "out.jsonl"
    error_path = tmp_path / "errors.jsonl"

    async def worker(item: str) -> dict:
        if item == "sample_bad":
            raise RuntimeError("boom")
        return {"sample_id": item}

    summary = await run_resumable_jobs(
        items=["sample_ok", "sample_bad"],
        item_id=lambda item: item,
        completed_id=lambda row: row.get("sample_id"),
        output_path=output_path,
        error_path=error_path,
        worker=worker,
        concurrency=2,
        max_attempts=2,
        retry_backoff_seconds=0,
        progress_label="test",
    )

    output_rows = list(read_jsonl(output_path))
    error_rows = list(read_jsonl(error_path))
    assert [row["sample_id"] for row in output_rows] == ["sample_ok"]
    assert error_rows[0]["item_id"] == "sample_bad"
    assert error_rows[0]["error_type"] == "RuntimeError"
    assert summary.succeeded == 1
    assert summary.failed == 1


@pytest.mark.anyio
async def test_run_resumable_jobs_prunes_resolved_errors(tmp_path) -> None:
    output_path = tmp_path / "out.jsonl"
    error_path = tmp_path / "errors.jsonl"
    append_jsonl(output_path, [{"sample_id": "sample_1"}])
    append_jsonl(
        error_path,
        [
            {"item_id": "sample_1", "error_type": "RuntimeError"},
            {"item_id": "sample_2", "error_type": "RuntimeError"},
        ],
    )

    async def worker(item: str) -> dict:
        return {"sample_id": item}

    summary = await run_resumable_jobs(
        items=["sample_1", "sample_2"],
        item_id=lambda item: item,
        completed_id=lambda row: row.get("sample_id"),
        output_path=output_path,
        error_path=error_path,
        worker=worker,
        retry_backoff_seconds=0,
        progress_label="test",
    )

    error_rows = list(read_jsonl(error_path))
    assert error_rows == []
    assert summary.completed_existing == 1
    assert summary.succeeded == 1
    assert summary.failed == 0
