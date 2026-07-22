import json
import re
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

package «lean_workbook_reprover» where
  buildType := .release
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩,
    ⟨`pp.proofs.withType, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.11.0"

@[default_target]
lean_lib «LeanWorkbookGen» where
"""
    lakefile = PROJECT_DIR / "lakefile.lean"
    lakefile.write_text(content, encoding="utf-8")

def write_lean_toolchain() -> None:
    content = "leanprover/lean4:v4.11.0\n"
    toolchain = PROJECT_DIR / "lean-toolchain"
    toolchain.write_text(content, encoding="utf-8")

def write_leanworkbookgen_root_lean(successful_files: list) -> None:
    lines = ["-- Auto-generated", ""]
    for split, stem in successful_files:
        lines.append(f"import LeanWorkbookGen.{split}.{stem}")
    root = PROJECT_DIR / "LeanWorkbookGen.lean"
    root.write_text("\n".join(lines) + "\n", encoding="utf-8")

# Matches the declaration name in a Lean theorem/lemma statement, e.g.
# "theorem lean_workbook_plus_317 (x : ...) ..." -> "lean_workbook_plus_317".
_THEOREM_DECL_RE = re.compile(r"\b(?:theorem|lemma)\s+([^\s({\[:]+)")


def _parse_theorem(content: str):
    """Return (theorem_name, 1-based line number) parsed from generated file text.

    The theorem name is taken from the actual ``theorem``/``lemma`` declaration
    (the source of truth that LeanDojo locates by fully-qualified name), not from
    the dataset ``id``, which is neither unique nor guaranteed to match.
    """
    for idx, line in enumerate(content.splitlines(), start=1):
        m = _THEOREM_DECL_RE.search(line)
        if m:
            return m.group(1), idx
    return None, None


import shutil

def main() -> None:
    from datasets import load_dataset
    # Start with a completely fresh project directory
    if PROJECT_DIR.exists():
        shutil.rmtree(PROJECT_DIR)
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading InternLM/Lean-Workbook dataset...")
    dataset = load_dataset("InternLM/Lean-Workbook")
    all_rows = list(dataset["train"])
    
    val_rows = all_rows[:100]
    test_rows = all_rows[100:200]

    # Pre-populate lakefile and toolchain for discovery builds
    write_lakefile_lean()
    write_lean_toolchain()
    
    # Write a dummy root file so lake discovery works
    (PROJECT_DIR / "LeanWorkbookGen.lean").touch()

    # Resolve dependencies and fetch prebuilt Mathlib oleans so we don't
    # compile all of Mathlib from source (which takes hours).
    print("Resolving dependencies (lake update)...")
    subprocess.run(["lake", "update"], cwd=PROJECT_DIR, check=True)
    print("Fetching prebuilt Mathlib cache (lake exe cache get)...")
    subprocess.run(["lake", "exe", "cache", "get"], cwd=PROJECT_DIR, check=True)

    # Rigorous discovery: write and build each file individually.
    # Records are collected here, keyed by the SAME original index `i` used for
    # the filename/namespace, so the JSON's file_path/full_name always match the
    # files on disk (the previous compaction step desynced these and made
    # LeanDojo discard theorems whose full_name didn't match the traced file).
    print("Rigorous build discovery (this may take a few minutes)...")
    successful_files = []
    records = {"val": [], "test": []}

    for split_prefix, rows in [("val", val_rows), ("test", test_rows)]:
        split_dir = PROJECT_DIR / "LeanWorkbookGen" / split_prefix
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for i, row in enumerate(rows):
            stem = f"lw_{split_prefix}_{i:04d}"
            module = f"LeanWorkbookGen.{split_prefix}.{stem}"
            lean_path = split_dir / f"{stem}.lean"
            
            # Write theorem to project for trial build
            statement = row.get("formal_statement", "")
            namespace = f"LW_{split_prefix}_{i:04d}"
            content = f"import Mathlib\n\nnamespace {namespace}\n\n{statement}\n\nend {namespace}"
            content = content.replace(":= by sorry", ":= sorry")
            lean_path.write_text(content, encoding="utf-8")
            
            # Try building this specific module
            res = subprocess.run(["lake", "build", module], cwd=PROJECT_DIR, capture_output=True)
            
            if res.returncode != 0:
                # If build fails, DELETE the file immediately so it doesn't pollute the project
                if lean_path.exists():
                    lean_path.unlink()
                print(f"Skipping {row['id']} (build failed)")
                continue

            theorem_name, theorem_line = _parse_theorem(content)
            if theorem_name is None:
                lean_path.unlink()
                print(f"Skipping {row['id']} (no theorem/lemma declaration found)")
                continue

            successful_files.append((split_prefix, stem))
            records[split_prefix].append({
                "id": row["id"],
                "file_path": f"LeanWorkbookGen/{split_prefix}/{stem}.lean",
                "full_name": f"{namespace}.{theorem_name}",
                "start": [theorem_line, 1],
                "split": split_prefix,
                "informal_stmt": row.get("natural_language_statement", ""),
                "informal_proof": row.get("answer", ""),
            })
            print(f"Verified {row['id']} -> {namespace}.{theorem_name}")

    # Write root module ONLY containing successful imports
    write_leanworkbookgen_root_lean(successful_files)
    
    # Final cleanup and build of valid subset to ensure zero exit code
    print("Cleaning and performing final build of valid subset...")
    subprocess.run(["lake", "clean"], cwd=PROJECT_DIR, check=True)
    subprocess.run(["lake", "build"], cwd=PROJECT_DIR, check=True)

    commit = _git_head(PROJECT_DIR)
    project_url = str(PROJECT_DIR.resolve())

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split_prefix in ("val", "test"):
        for rec in records[split_prefix]:
            rec["url"] = project_url
            rec["commit"] = commit
        with open(DATA_DIR / "lean_workbook_reprover" / f"{split_prefix}.json", "w") as f:
            json.dump(records[split_prefix], f, indent=2)

    print(f"Done. Project at {PROJECT_DIR}")

if __name__ == "__main__":
    main()
