"""Clean up dojo_full run logs and compute per-K x R summary statistics.

The raw logs in data/dojo_full/*.log are written by one or more worker
processes without a shared file lock. This creates two problems that must be
cleaned up before the data can be trusted:

1. At higher concurrency (larger K, R) multiple workers' output interleaves
   at the byte level within a single line, producing garbled "PID:<partial
   digits>" tokens that only occur a handful of times. Real workers write
   thousands of lines each, so we drop any PID with a line count below
   ``--min-worker-lines`` as corruption noise.
2. A theorem can appear more than once in a log (crash + resume). We keep
   only the LAST recorded SearchResult for each theorem, since it is the
   authoritative final attempt.

Usage (run with cwd = the paper repo root, i.e. one level above `code/`):
    python code/scripts/clean_dojo_logs.py --data-dir data/leandojo_sweep/raw_logs \
        --out data/leandojo_sweep/dojo_full_clean.json

    python code/scripts/clean_dojo_logs.py --data-dir data/leandojo_sweep/raw_logs \
        --out data/leandojo_sweep/dojo_full_clean.json --table
"""
import re
import json
import glob
import os
import argparse
from collections import defaultdict, Counter

PID_RE = re.compile(r"PID:(\d+)")
NAME_RE = re.compile(r"full_name=(?:'([^']*)'|\"([^\"]*)\")")
STATUS_RE = re.compile(r"status=<Status\.(\w+):")
TIME_RE = re.compile(r"total_time=([0-9.eE+-]+)")


def real_pids(path, min_lines):
    counts = Counter()
    with open(path, errors="replace") as fh:
        for line in fh:
            m = PID_RE.search(line)
            if m:
                counts[m.group(1)] += 1
    return {pid for pid, c in counts.items() if c >= min_lines}, counts


def parse_log(path, min_worker_lines):
    keep_pids, pid_counts = real_pids(path, min_worker_lines)
    n_corrupt_lines_dropped = sum(c for pid, c in pid_counts.items() if pid not in keep_pids)

    results = {}
    current_name = defaultdict(lambda: None)
    pending_result = defaultdict(lambda: True)
    n_proving = 0
    n_searchresult = 0
    n_duplicate_theorems = 0
    n_crashed = 0
    seen_names = set()

    with open(path, errors="replace") as fh:
        for line in fh:
            pm = PID_RE.search(line)
            if not pm or pm.group(1) not in keep_pids:
                continue
            pid = pm.group(1)

            if "INFO | Proving Theorem" in line:
                if current_name[pid] is not None and not pending_result[pid]:
                    n_crashed += 1
                m = NAME_RE.search(line)
                if not m:
                    continue
                name = m.group(1) or m.group(2)
                if name in seen_names:
                    n_duplicate_theorems += 1
                seen_names.add(name)
                current_name[pid] = name
                pending_result[pid] = False
                n_proving += 1
            elif "INFO | SearchResult" in line:
                m = NAME_RE.search(line)
                if not m:
                    continue
                name = m.group(1) or m.group(2)
                sm = STATUS_RE.search(line)
                tm = TIME_RE.search(line)
                status = sm.group(1) if sm else None
                total_time = float(tm.group(1)) if tm else None
                n_searchresult += 1
                results[name] = {"status": status, "total_time": total_time}
                pending_result[pid] = True

    for pid, name in current_name.items():
        if name is not None and not pending_result[pid]:
            n_crashed += 1

    stats = {
        "n_workers_kept": len(keep_pids),
        "n_corrupt_lines_dropped": n_corrupt_lines_dropped,
        "n_proving_lines": n_proving,
        "n_searchresult_lines": n_searchresult,
        "n_unique_theorems": len(results),
        "n_duplicate_reattempts": n_duplicate_theorems,
        "n_crashed_no_result": n_crashed,
    }
    return results, stats


def proved(rec):
    return rec is not None and rec["status"] == "PROVED"


CONFIGS = {
    1: {0: "thm2000_tac1_norepair.log", 1: "thm2000_tac1_repair_c1.log",
        2: "thm2000_tac1_repair_c2.log", 3: "thm2000_tac1_repair_c3.log"},
    3: {0: "thm2000_tac3_norepair.log", 1: "thm2000_tac3_repair_c1.log",
        2: "thm2000_tac3_repair_c2.log", 3: "thm2000_tac3_repair_c3.log"},
    5: {0: "thm2000_tac5_norepair.log", 1: "thm2000_tac5_repair_c1.log",
        2: "thm2000_tac5_repair_c2.log", 3: "thm2000_tac5_repair_c3.log"},
    8: {0: "thm2000_tac8_norepair.log", 1: "thm2000_tac8_repair_c1.log",
        2: "thm2000_tac8_repair_c2.log", 3: "thm2000_tac8_repair_c3.log"},
}


def build_table(all_results):
    # Use one global common set of theorems across every K x R run so that the
    # attempted count (and every derived rate) is identical for all budgets and
    # the columns are compared over the same theorems. Without this, each K's
    # own intersection differs (e.g. K=8 attempts a slightly larger set), which
    # makes the per-K pass rates not directly comparable.
    common = set.intersection(
        *(set(all_results[fn]["results"].keys())
          for files in CONFIGS.values() for fn in files.values())
    )
    atmp = len(common)
    rows = []
    for K, files in CONFIGS.items():
        results_by_r = {r: all_results[fn]["results"] for r, fn in files.items()}
        base = results_by_r[0]
        base_pass = sum(1 for n in common if proved(base[n]))
        p1b = 100.0 * base_pass / atmp
        for r in (0, 1, 2, 3):
            rep = results_by_r[r]
            rep_pass = sum(1 for n in common if proved(rep[n]))
            p1r = 100.0 * rep_pass / atmp
            fix = sum(1 for n in common if not proved(base[n]) and proved(rep[n]))
            brk = sum(1 for n in common if proved(base[n]) and not proved(rep[n]))
            net = fix - brk
            rows.append(dict(K=K, R=r, Atmp=atmp, P1B=p1b, P1R=p1r, Fix=fix, Brk=brk, Net=net))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/leandojo_sweep/raw_logs")
    ap.add_argument("--out", default="data/leandojo_sweep/dojo_full_clean.json")
    ap.add_argument("--min-worker-lines", type=int, default=200,
                     help="PIDs with fewer log lines than this are treated as corruption noise")
    ap.add_argument("--table", action="store_true", help="print the K x R summary table")
    args = ap.parse_args()

    out = {}
    for path in sorted(glob.glob(os.path.join(args.data_dir, "*.log"))):
        fname = os.path.basename(path)
        results, stats = parse_log(path, args.min_worker_lines)
        out[fname] = {"stats": stats, "results": results}
        print(f"{fname}: {stats}")

    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"Wrote {args.out}")

    if args.table:
        rows = build_table(out)
        print(f"\n{'K':>2} {'R':>2} {'Atmp':>6} {'P@1':>7} {'Fix':>4} {'Brk':>4} {'Net':>5}")
        for row in rows:
            print(f"{row['K']:>2} {row['R']:>2} {row['Atmp']:>6} {row['P1R']:>7.1f} "
                  f"{row['Fix']:>4} {row['Brk']:>4} {row['Net']:>+5d}")


if __name__ == "__main__":
    main()
