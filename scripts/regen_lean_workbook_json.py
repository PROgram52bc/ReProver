"""Regenerate data/lean_workbook_reprover/{val,test}.json from the already-built
project files, without rebuilding or re-tracing.

The original setup desynced the JSON indices from the on-disk filenames, so the
recorded ``full_name`` pointed at the wrong theorem and LeanDojo discarded it.
This reads each generated ``lw_<split>_<i>.lean`` and rebuilds the record from
the file's actual namespace + theorem name, mapping informal fields back to the
dataset row by the file's original index ``i``.
"""

import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROJECT_DIR = DATA_DIR / "lean_workbook_reprover" / "project"

_THEOREM_DECL_RE = re.compile(r"\b(?:theorem|lemma)\s+([^\s({\[:]+)")
_NAMESPACE_RE = re.compile(r"^namespace\s+(\S+)", re.MULTILINE)
_STEM_RE = re.compile(r"_(\d+)$")


def _parse_file(text: str):
    ns_m = _NAMESPACE_RE.search(text)
    namespace = ns_m.group(1) if ns_m else None
    for idx, line in enumerate(text.splitlines(), start=1):
        m = _THEOREM_DECL_RE.search(line)
        if m:
            return namespace, m.group(1), idx
    return namespace, None, None


def main() -> None:
    from datasets import load_dataset

    dataset = load_dataset("InternLM/Lean-Workbook")
    all_rows = list(dataset["train"])
    rows_by_split = {"val": all_rows[:100], "test": all_rows[100:200]}

    commit = subprocess.check_output(
        ["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"], text=True
    ).strip()
    project_url = str(PROJECT_DIR.resolve())

    for split in ("val", "test"):
        split_dir = PROJECT_DIR / "LeanWorkbookGen" / split
        rows = rows_by_split[split]
        records = []
        for path in sorted(split_dir.glob(f"lw_{split}_*.lean")):
            stem = path.stem
            m = _STEM_RE.search(stem)
            if not m:
                continue
            i = int(m.group(1))
            namespace, theorem_name, theorem_line = _parse_file(
                path.read_text(encoding="utf-8")
            )
            if theorem_name is None or namespace is None:
                print(f"WARN: could not parse theorem in {path.name}; skipping")
                continue
            row = rows[i]
            records.append({
                "id": row["id"],
                "file_path": f"LeanWorkbookGen/{split}/{stem}.lean",
                "full_name": f"{namespace}.{theorem_name}",
                "start": [theorem_line, 1],
                "split": split,
                "informal_stmt": row.get("natural_language_statement", ""),
                "informal_proof": row.get("answer", ""),
                "url": project_url,
                "commit": commit,
            })
        out = DATA_DIR / "lean_workbook_reprover" / f"{split}.json"
        out.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
