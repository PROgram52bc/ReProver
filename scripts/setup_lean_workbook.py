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

def write_leanworkbookgen_root_lean(successful_files: list) -> None:
    lines = ["-- Auto-generated", ""]
    for split, stem in successful_files:
        lines.append(f"import LeanWorkbookGen.{split}.{stem}")
    root = PROJECT_DIR / "LeanWorkbookGen.lean"
    root.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _build_records(rows: list, split_prefix: str, split_label: str) -> list:
    project_url = str(PROJECT_DIR.resolve())
    commit = _git_head(PROJECT_DIR)
    records = []

    # We only want to include theorems that actually build, because LeanDojo
    # fails if 'lake build' has a non-zero exit code.
    print(f"Filtering successful builds for {split_label}...")
    for i, row in enumerate(rows):
        pid = row["id"]
        # The actual name of the theorem is usually the last part of the pid
        # or we can extract it if we want to be precise, but since we wrapped
        # it in a namespace LW_split_index, the full name for LeanDojo
        # should be LW_split_index.<theorem_name>
        # Looking at InternLM/Lean-Workbook, the formal_statement usually looks like
        # 'theorem lean_workbook_plus_317 ...'
        theorem_name = pid.split('.')[-1]
        namespace = f"LW_{split_prefix}_{i:04d}"
        full_name = f"{namespace}.{theorem_name}"
        
        file_path = f"LeanWorkbookGen/{split_prefix}/lw_{split_prefix}_{i:04d}.lean"

        # Check if the .olean exists (which means it built successfully)
        # Lake builds into .lake/build/lib/LeanWorkbookGen/...
        olean_path = PROJECT_DIR / ".lake" / "build" / "lib" / "LeanWorkbookGen" / split_prefix / f"lw_{split_prefix}_{i:04d}.olean"

        if not olean_path.exists():
            print(f"Skipping {pid} due to build failure.")
            continue

        records.append({
            "id": pid,
            "file_path": file_path,
            "full_name": full_name,
            "start": [3, 1], # Usually starts on line 3 after imports
            "split": split_label,
            "informal_stmt": row.get("natural_language_statement", ""),
            "informal_proof": row.get("answer", ""),
            "url": project_url,
            "commit": commit,
        })
    return records


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

    # Rigorous discovery: write and build each file individually
    print("Rigorous build discovery (this may take a few minutes)...")
    successful_files = []
    
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
            
            if res.returncode == 0:
                successful_files.append((split_prefix, stem))
                print(f"Verified {row['id']}")
            else:
                # If build fails, DELETE the file immediately so it doesn't pollute the project
                if lean_path.exists():
                    lean_path.unlink()
                print(f"Skipping {row['id']} (build failed)")

    # Write root module ONLY containing successful imports
    write_leanworkbookgen_root_lean(successful_files)
    
    # Final cleanup and build of valid subset to ensure zero exit code
    print("Cleaning and performing final build of valid subset...")
    subprocess.run(["lake", "clean"], cwd=PROJECT_DIR, check=True)
    subprocess.run(["lake", "build"], cwd=PROJECT_DIR, check=True)

    _git_head(PROJECT_DIR)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Update indices to only include verified successful theorems
    successful_stems = {f"{s}.{t}" for s, t in successful_files}
    
    verified_val = []
    for i, row in enumerate(val_rows):
        if f"val.lw_val_{i:04d}" in successful_stems:
            verified_val.append(row)
            
    verified_test = []
    for i, row in enumerate(test_rows):
        if f"test.lw_test_{i:04d}" in successful_stems:
            verified_test.append(row)

    with open(DATA_DIR / "lw_val.json", "w") as f:
        json.dump(_build_records(verified_val, "val", "val"), f, indent=2)
    with open(DATA_DIR / "lw_test.json", "w") as f:
        json.dump(_build_records(verified_test, "test", "test"), f, indent=2)

    print(f"Done. Project at {PROJECT_DIR}")

if __name__ == "__main__":
    main()
