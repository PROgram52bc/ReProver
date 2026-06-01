import argparse
import csv
import os

from prover.attempt_summary import read_jsonl, replay_global_budget


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay theorem attempt summaries under multiple timeout accounting views."
    )
    parser.add_argument("summaries", nargs="+", help="JSONL files from prover/evaluate.py --summary-jsonl.")
    parser.add_argument("--local-timeout", type=float, default=600.0)
    parser.add_argument("--global-budget", type=float, default=100000.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for summary in args.summaries:
        records = read_jsonl(summary)
        run_name = os.path.basename(summary).removesuffix(".jsonl")
        for mode in ("wall", "effective"):
            row = replay_global_budget(
                records,
                timeout=args.local_timeout,
                mode=mode,
                global_budget=args.global_budget,
            )
            row["run"] = run_name
            row["summary"] = summary
            rows.append(row)

    fieldnames = [
        "run",
        "summary",
        "timeout_accounting",
        "local_timeout",
        "global_budget",
        "elapsed",
        "attempted",
        "proved",
        "failed",
        "timed_out",
        "discarded",
        "proofs_per_hour",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
