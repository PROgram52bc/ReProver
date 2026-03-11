"""
LeanDojo Log Analyzer
Handles both multi-line (BEST) and compact single-line (BFS/DFS) STEP formats.
"""

import re
from collections import Counter
from pathlib import Path

LOG_FILES = [
    ("./logs/BEST-Retrieval-5-200.log", "BEST-Retrieval-5-200"),
    # ("./logs/BFS-retrieval-5-50.log",   "BFS-Retrieval-5-50"),
    # ("./logs/DFS-retrieval-5-50.log",   "DFS-Retrieval-5-50"),
]

# Two regex variants for the two log formats
STEP_MULTI = re.compile(r"=== STEP ===\n\[THEOREM\]: (.+?)\n\[STATE\]:\n(.*?)\n\[TACTIC\]: (.+?)\n\[RESULT \((\w+)\)\]:\n(.*?)\n={4,}", re.DOTALL)

STEP_COMPACT = re.compile(r"=== STEP ===\[THEOREM\]: (.+?)\[STATE\]: (.*?)\[TACTIC\]: (.+?)\[RESULT \((\w+)\)\]: (.*?)={4,}", re.DOTALL)

THEOREM_RE = re.compile(r"INFO \| Proving Theorem\(.*?full_name='(.+?)'\)")

ERROR_PATTERNS = [
    ("rfl_failed",         r"rfl tactic failed"),
    ("rewrite_failed",     r"tactic 'rewrite' failed"),
    ("unknown_identifier", r"unknown identifier|unknown constant"),
    ("type_mismatch",      r"type mismatch"),
    ("subst_failed",       r"tactic 'subst' failed"),
    ("rcases_failed",      r"rcases tactic failed"),
    ("intro_failed",       r"tactic 'intro"),
    ("apply_failed",       r"tactic 'apply' failed"),
    ("exact_failed",       r"tactic 'exact' failed"),
    ("simp_failed",        r"simp.*failed"),
    ("not_inductive",      r"is not an inductive datatype"),
    ("timeout",            r"timeout|Timeout"),
    ("other_error",        r".+"),
]

# categorize the errors. if not in list, add as other_error
def categorize(error_text):
    for label, pat in ERROR_PATTERNS:
        if re.search(pat, error_text, re.IGNORECASE):
            return label
    return "other_error"


# extract text from the log file. We'll use the STEP_COMPACT regex.
def parse_log(path):
    text = Path(path).read_text(errors="replace")
    # Try multi-line format first
    steps = [{"theorem": m[0].strip(), "state": m[1].strip(),
               "tactic": m[2].strip(), "result_type": m[3].strip(),
               "result_body": m[4].strip()}
             for m in STEP_MULTI.findall(text)]
    if not steps:
        steps = [{"theorem": m[0].strip(), "state": m[1].strip(),
                   "tactic": m[2].strip(), "result_type": m[3].strip(),
                   "result_body": m[4].strip()}
                 for m in STEP_COMPACT.findall(text)]
    theorems = THEOREM_RE.findall(text)
    return steps, theorems

# tactic is accepted but next step also same state and theorem, indicating a stall (no progress)
def find_stalls(steps):
    stalls = []
    for i in range(len(steps) - 1):
        s, nxt = steps[i], steps[i+1]
        if (s["result_type"] == "TacticState"
                and s["theorem"] == nxt["theorem"]
                and s["state"] == nxt["state"]):
            stalls.append(s)
    return stalls

# Analyze a single log file and print summary statistics and examples
def analyze(path, label):
    steps, theorems = parse_log(path)
    errors = [s for s in steps if s["result_type"] == "LeanError"]
    success = [s for s in steps if s["result_type"] == "ProofFinished"]
    stalls  = find_stalls(steps)

    outcome_counts = Counter(s["result_type"] for s in steps)
    cat_counts     = Counter(categorize(s["result_body"]) for s in errors)
    fail_tactics   = Counter(s["tactic"] for s in errors)
    stall_tactics  = Counter(s["tactic"] for s in stalls)

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  LOG: {label}")
    print(sep)
    print(f"  Theorems attempted : {len(theorems)}")
    print(f"  Total tactic steps : {len(steps)}")

    print(f"\n  Tactic outcomes:")
    for outcome, cnt in outcome_counts.most_common():
        pct = 100 * cnt / len(steps) if steps else 0
        print(f"    {outcome:<20} {cnt:>5}  ({pct:.1f}%)")

    if errors:
        print(f"\n  Error category breakdown ({len(errors)} errors total):")
        for cat, cnt in cat_counts.most_common():
            pct = 100 * cnt / len(errors)
            print(f"    {cat:<25} {cnt:>5}  ({pct:.1f}%)")

        print(f"\n  Top 10 failing tactics:")
        for tac, cnt in fail_tactics.most_common(10):
            tac_short = tac.replace("\n", " ")[:60]
            print(f"    {cnt:>4}x  {tac_short}")

    print(f"\n  Stalls (accepted, state unchanged): {len(stalls)}")
    if stall_tactics:
        print(f"  Top 5  stalling tactics:")
        for tac, cnt in stall_tactics.most_common(5):
            print(f"    {cnt:>4}x  {tac}")

    # # Concrete examples — one per category, real data only
    # print(f"\n  {'─'*60}")
    # print(f"  CONCRETE FAILURE EXAMPLES (real log data)")
    # print(f"  {'─'*60}")
    # shown = set()
    # for s in errors:
    #     cat = categorize(s["result_body"])
    #     if cat in shown:
    #         continue
    #     shown.add(cat)
    #     state_lines = s["state"].splitlines()
    #     state_preview = "\n      ".join(state_lines[:5])
    #     if len(state_lines) > 5:
    #         state_preview += f"\n      ... ({len(state_lines)-5} more lines)"
    #     err_preview = s["result_body"].splitlines()[0][:110]
    #     print(f"\n  [{cat}]")
    #     print(f"    Theorem : {s['theorem']}")
    #     print(f"    State   :\n      {state_preview}")
    #     print(f"    Tactic  : {s['tactic'].replace(chr(10),' ')[:80]}")
    #     print(f"    Error   : {err_preview}")
    #     if len(shown) >= 5:
    #         break

    # # One stall example
    # if stalls:
    #     s = stalls[0]
    #     state_lines = s["state"].splitlines()
    #     goal_line = next((l for l in state_lines if l.startswith("⊢")), state_lines[-1])
    #     print(f"\n  [stall — accepted, no progress]")
    #     print(f"    Theorem : {s['theorem']}")
    #     print(f"    Goal    : {goal_line}")
    #     print(f"    Tactic  : {s['tactic']}")
    #     print(f"    Result  : TacticState accepted, but next step has identical proof state")

    return {
        "label": label, "theorems": len(theorems), "steps": len(steps),
        "errors": len(errors), "success": len(success), "stalls": len(stalls),
        "cat_counts": cat_counts, "stall_tactics": stall_tactics,
    }

# ── Run ───────────────────────────────────────────────────────────────────────
all_results = []
for path, label in LOG_FILES:
    r = analyze(path, label)
    all_results.append(r)

print(f"\n\n{'='*72}")
print("  CROSS-LOG SUMMARY")
print(f"{'='*72}")
print(f"  {'Log':<28} {'Thms':>5} {'Steps':>6} {'Errors':>7} {'Err%':>6} {'Stalls':>7}")
print(f"  {'-'*28} {'-'*5} {'-'*6} {'-'*7} {'-'*6} {'-'*7}")
for r in all_results:
    pct = 100 * r["errors"] / r["steps"] if r["steps"] else 0
    print(f"  {r['label']:<28} {r['theorems']:>5} {r['steps']:>6} {r['errors']:>7} {pct:>5.1f}% {r['stalls']:>7}")

print("\nDone.")