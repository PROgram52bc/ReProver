from typing import Literal

TimeoutAccounting = Literal["wall", "effective"]


def charged_time(total_time: float, repair_time: float, mode: TimeoutAccounting) -> float:
    """Return the time charged against a local per-theorem budget.

    Under 'wall' accounting, all elapsed time counts.
    Under 'effective' accounting, measured repair overhead is subtracted.
    """
    if mode == "wall":
        return total_time
    if mode == "effective":
        return max(0.0, total_time - repair_time)
    raise ValueError(f"Unknown timeout accounting mode: {mode!r}")


def should_stop_local(
    total_time: float,
    repair_time: float,
    timeout: float,
    mode: TimeoutAccounting,
) -> bool:
    """Return True when the charged time exceeds the local per-theorem budget."""
    return charged_time(total_time, repair_time, mode) > timeout
