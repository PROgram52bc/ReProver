"""Compare a baseline (no-repair) and a repair run from their --summary-jsonl outputs.

Reads two JSONL files written by ``prover/evaluate.py --summary-jsonl`` (one
run without --repair-ckpt-path, one with it) over the *same* theorem set, and
reports Pass@1 for each plus which theorems flipped between runs.

Only theorems present in a run's JSONL count towards its Pass@1 -- discarded
("non-theorem") attempts are never written to summary_jsonl by evaluate.py,
so they are correctly excluded here too, matching evaluate.py's own
num_proved / (num_proved + num_failed) calculation.
"""

import argparse

from prover.attempt_summary import AttemptRecord, read_jsonl


def pass_at_1(records: list[AttemptRecord]) -> float:
    if not records:
        return float("nan")
    proved = sum(1 for r in records if r.status == "Proved")
    return proved / len(records)


def summarize(name: str, records: list[AttemptRecord]) -> None:
    proved = sum(1 for r in records if r.status == "Proved")
    print(f"{name:12} attempted={len(records):3d}  proved={proved:3d}  Pass@1={pass_at_1(records):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_jsonl", help="--summary-jsonl output from the no-repair run")
    parser.add_argument("repair_jsonl", help="--summary-jsonl output from the repair run")
    args = parser.parse_args()

    baseline = {r.theorem: r for r in read_jsonl(args.baseline_jsonl)}
    repair = {r.theorem: r for r in read_jsonl(args.repair_jsonl)}

    print("=" * 60)
    print("BASELINE vs REPAIR")
    print("=" * 60)
    summarize("baseline", list(baseline.values()))
    summarize("repair", list(repair.values()))
    print()

    common = sorted(set(baseline) & set(repair))
    only_baseline = sorted(set(baseline) - set(repair))
    only_repair = sorted(set(repair) - set(baseline))
    if only_baseline or only_repair:
        print(
            f"NOTE: theorem sets differ (baseline-only: {len(only_baseline)}, "
            f"repair-only: {len(only_repair)}) -- comparison below is restricted "
            f"to the {len(common)} theorems attempted in both runs.\n"
        )

    gained = [t for t in common if baseline[t].status != "Proved" and repair[t].status == "Proved"]
    lost = [t for t in common if baseline[t].status == "Proved" and repair[t].status != "Proved"]
    both_proved = [t for t in common if baseline[t].status == "Proved" and repair[t].status == "Proved"]

    print(f"Proved by both:            {len(both_proved)}")
    print(f"Proved by repair only:     {len(gained)}")
    for t in gained:
        print(f"  + {t}")
    print(f"Proved by baseline only:   {len(lost)}")
    for t in lost:
        print(f"  - {t}")

    repaired_hits = [repair[t] for t in common if repair[t].status == "Proved" and repair[t].repair_time > 0]
    if repaired_hits:
        avg_repair_time = sum(r.repair_time for r in repaired_hits) / len(repaired_hits)
        print(
            f"\n{len(repaired_hits)} of the repair run's proofs actually invoked "
            f"repair (repair_time > 0); average repair_time for those = {avg_repair_time:.2f}s"
        )


if __name__ == "__main__":
    main()
