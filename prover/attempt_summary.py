from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Iterable, Literal

from prover.timing import TimeoutAccounting, charged_time

LocalOutcome = Literal["Proved", "Failed", "Timeout", "Discarded"]


@dataclass(frozen=True)
class AttemptRecord:
    theorem: str
    status: str
    total_time: float
    repair_time: float
    actor_time: float
    environment_time: float
    num_total_nodes: int
    num_searched_nodes: int


def classify_local(record: AttemptRecord, timeout: float, mode: TimeoutAccounting) -> LocalOutcome:
    """Classify a theorem attempt under the given accounting mode and local timeout."""
    if record.status == "Discarded":
        return "Discarded"
    if charged_time(record.total_time, record.repair_time, mode) > timeout:
        return "Timeout"
    if record.status == "Proved":
        return "Proved"
    return "Failed"


def local_cost(record: AttemptRecord, timeout: float, mode: TimeoutAccounting) -> float:
    """Return the wall-clock cost charged for this attempt under the given accounting mode."""
    if record.status == "Discarded":
        return 0.0
    return min(charged_time(record.total_time, record.repair_time, mode), timeout)


def replay_global_budget(
    records: Iterable[AttemptRecord],
    timeout: float,
    mode: TimeoutAccounting,
    global_budget: float,
) -> dict[str, float | int | str]:
    """Replay a list of attempt records under a global wall-clock budget.

    Returns a summary dict with counts and a throughput metric.
    """
    elapsed = 0.0
    attempted = proved = failed = timed_out = discarded = 0

    for record in records:
        outcome = classify_local(record, timeout, mode)
        cost = local_cost(record, timeout, mode)
        if elapsed + cost > global_budget:
            break
        elapsed += cost

        if outcome == "Discarded":
            discarded += 1
            continue

        attempted += 1
        if outcome == "Proved":
            proved += 1
        elif outcome == "Timeout":
            timed_out += 1
        else:
            failed += 1

    return {
        "timeout_accounting": mode,
        "local_timeout": timeout,
        "global_budget": global_budget,
        "elapsed": elapsed,
        "attempted": attempted,
        "proved": proved,
        "failed": failed,
        "timed_out": timed_out,
        "discarded": discarded,
        "proofs_per_hour": 0.0 if elapsed == 0 else proved / (elapsed / 3600.0),
    }


def write_jsonl(path: str, records: Iterable[AttemptRecord]) -> None:
    """Write attempt records to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def read_jsonl(path: str) -> list[AttemptRecord]:
    """Read attempt records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(AttemptRecord(**json.loads(line)))
    return records
