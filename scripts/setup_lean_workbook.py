import json
import subprocess
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PROJECT_DIR = DATA_DIR / "lean_workbook_reprover" / "project"

def _git_head(repo: Path) -> str:
    if not (repo / ".git").exists():
        subprocess.run(["git", "init"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "you@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Your Name"], cwd=repo, check=True)
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-m", "initial commit"], cwd=repo, check=True)
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()

def write_lakefile_lean() -> None:
    content = """import Lake
open Lake DSL

package "lean_workbook_reprover" where
  version := v!"0.1.0"
  keywords := #["math"]
  buildType := .release
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩,
    ⟨`pp.proofs.withType, false⟩,
    ⟨`weak.ast, true⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.12.0"

@[default_target]
lean_lib «LeanWorkbookGen» where
  buildType := .debug
"""
    lakefile = PROJECT_DIR / "lakefile.lean"
    lakefile.write_text(content, encoding="utf-8")

def write_lean_toolchain() -> None:
    content = "leanprover/lean4:v4.12.0\n"
    toolchain = PROJECT_DIR / "lean-toolchain"
    toolchain.write_text(content, encoding="utf-8")

def write_leanworkbookgen_root_lean() -> None:
    lines = ["-- Auto-generated", ""]
    for split in ("val", "test"):
        d = PROJECT_DIR / "LeanWorkbookGen" / split
        if not d.is_dir(): continue
        for f in sorted(d.glob("lw_*.lean")):
            lines.append(f"import LeanWorkbookGen.{split}.{f.stem}")
    root = PROJECT_DIR / "LeanWorkbookGen.lean"
    root.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _build_records(rows: list, split_prefix: str, split_label: str) -> list:
    project_url = str(PROJECT_DIR.resolve())
    commit = _git_head(PROJECT_DIR)
    records = []
    for i, row in enumerate(rows):
        pid = row["id"]
        file_path = f"LeanWorkbookGen/{split_prefix}/lw_{split_prefix}_{i:04d}.lean"
        records.append({
            "id": pid,
            "file_path": file_path,
            "full_name": pid,
            "start": [1, 1],
            "split": split_label,
            "informal_stmt": row.get("natural_language_statement", ""),
            "informal_proof": row.get("answer", ""),
            "url": project_url,
            "commit": commit,
        })
    return records

def main() -> None:
    from datasets import load_dataset
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading InternLM/Lean-Workbook dataset...")
    dataset = load_dataset("InternLM/Lean-Workbook")
    all_rows = list(dataset["train"])
    
    val_rows = all_rows[:100]
    test_rows = all_rows[100:200]

    for split_prefix, rows in [("val", val_rows), ("test", test_rows)]:
        split_dir = PROJECT_DIR / "LeanWorkbookGen" / split_prefix
        split_dir.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(rows):
            path = split_dir / f"lw_{split_prefix}_{i:04d}.lean"
            statement = row.get("formal_statement", "")
            content = f"import Mathlib\n\n{statement}"
            content = content.replace(":= by sorry", ":= sorry")
            path.write_text(content, encoding="utf-8")

    write_lakefile_lean()
    write_lean_toolchain()
    write_leanworkbookgen_root_lean()

    _git_head(PROJECT_DIR) # Ensure git initialized and commit made

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "lw_val.json", "w") as f:
        json.dump(_build_records(val_rows, "val", "val"), f, indent=2)
    with open(DATA_DIR / "lw_test.json", "w") as f:
        json.dump(_build_records(test_rows, "test", "test"), f, indent=2)

    print(f"Done. Project at {PROJECT_DIR}")

if __name__ == "__main__":
    main()
