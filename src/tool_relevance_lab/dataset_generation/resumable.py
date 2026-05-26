from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel
from tqdm import tqdm

from tool_relevance_lab.dataset_generation.jsonl import append_jsonl, read_jsonl


ItemT = TypeVar("ItemT")
RowT = TypeVar("RowT", bound=BaseModel | dict)


@dataclass(frozen=True)
class FailedJobRecord:
    item_id: str
    error_type: str
    error_message: str
    attempt: int
    timestamp: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ResumableRunSummary:
    total: int
    completed_existing: int
    submitted: int
    succeeded: int
    failed: int
    skipped: int
    output_path: str
    error_path: str

    def to_dict(self) -> dict:
        return asdict(self)


async def run_resumable_jobs(
    *,
    items: Iterable[ItemT],
    item_id: Callable[[ItemT], str],
    completed_id: Callable[[dict], str | None],
    output_path: Path,
    error_path: Path,
    worker: Callable[[ItemT], Awaitable[RowT]],
    concurrency: int = 8,
    max_attempts: int = 3,
    retry_backoff_seconds: float = 1.0,
    progress_label: str = "jobs",
) -> ResumableRunSummary:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    item_list = list(items)
    completed = _completed_ids(output_path, completed_id)
    pending = [item for item in item_list if item_id(item) not in completed]
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    succeeded = 0
    failed = 0

    async def run_one(item: ItemT) -> None:
        current_id = item_id(item)
        async with semaphore:
            for attempt in range(1, max_attempts + 1):
                try:
                    row = await worker(item)
                    return row, None
                except Exception as exc:  # noqa: BLE001 - persist job failure context
                    if attempt < max_attempts:
                        await asyncio.sleep(retry_backoff_seconds * attempt)
                        continue
                    error = FailedJobRecord(
                        item_id=current_id,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        attempt=attempt,
                        timestamp=time.time(),
                    )
                    return None, error
        raise RuntimeError(f"job did not complete: {current_id}")

    progress = tqdm(total=len(item_list), initial=len(completed), desc=progress_label, unit="job")
    try:
        tasks = [asyncio.create_task(run_one(item)) for item in pending]
        for task in asyncio.as_completed(tasks):
            row, error = await task
            async with lock:
                if row is not None:
                    append_jsonl(output_path, [row])
                    succeeded += 1
                elif error is not None:
                    append_jsonl(error_path, [error.to_dict()])
                    failed += 1
            progress.update(1)
            progress.set_postfix(done=len(completed) + succeeded + failed, failed=failed)
    finally:
        progress.close()

    return ResumableRunSummary(
        total=len(item_list),
        completed_existing=len(completed),
        submitted=len(pending),
        succeeded=succeeded,
        failed=failed,
        skipped=len(completed),
        output_path=str(output_path),
        error_path=str(error_path),
    )


def _completed_ids(path: Path, completed_id: Callable[[dict], str | None]) -> set[str]:
    ids: set[str] = set()
    for row in read_jsonl(path):
        value = completed_id(row)
        if value is not None:
            ids.add(value)
    return ids


def write_run_summary(path: Path, summary: ResumableRunSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
