import json
import importlib
import os
import re
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MINIF2F_DIR = DATA_DIR / "minif2f"
PROJECT_DIR = MINIF2F_DIR / "project"
GEN_ROOT = PROJECT_DIR / "MiniF2FGen"
LEAN_TOOLCHAIN = "leanprover/lean4:v4.12.0"
MATHLIB_COMMIT = "v4.12.0"


def _load_raw_splits() -> tuple[list[dict], list[dict]]:
    val_path = DATA_DIR / "val.json"
    test_path = DATA_DIR / "test.json"
    if val_path.exists() and test_path.exists():
        return json.loads(val_path.read_text()), json.loads(test_path.read_text())

    datasets_module = importlib.import_module("datasets")
    dataset = datasets_module.load_dataset("cat-searcher/minif2f-lean4")
    split_name = "validation" if "validation" in dataset else "valid"
    val_rows = list(dataset[split_name])
    test_rows = list(dataset["test"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    val_path.write_text(json.dumps(val_rows, indent=2))
    test_path.write_text(json.dumps(test_rows, indent=2))
    return val_rows, test_rows


def _write_project_files() -> None:
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_DIR / ".gitignore").write_text(".lake/\n")
    (PROJECT_DIR / "lean-toolchain").write_text(f"{LEAN_TOOLCHAIN}\n")
    (PROJECT_DIR / "lakefile.lean").write_text(
        (
            "import Lake\n"
            "open Lake DSL\n\n"
            'package "minif2f_reprover" where\n'
            "  buildType := .release\n\n"
            "require mathlib from git\n"
            f'  "https://github.com/leanprover-community/mathlib4.git" @ "{MATHLIB_COMMIT}"\n\n'
            "@[default_target]\n"
            "lean_lib MiniF2FGen\n"
        )
    )


def _parse_theorem_name(formal_statement: str, fallback: str) -> str:
    match = re.search(r"(?m)^\\s*theorem\\s+([A-Za-z0-9_'.]+)", formal_statement)
    return match.group(1) if match else fallback.replace("-", "_")


def _write_split(split: str, rows: list[dict]) -> list[dict]:
    split_module = "Val" if split == "val" else "Test"
    split_dir = GEN_ROOT / split_module
    split_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for idx, row in enumerate(rows):
        stem = f"mf2f_{split}_{idx:04d}"
        namespace = f"MiniF2F_{split_module}_{idx:04d}"
        theorem_name = _parse_theorem_name(row["formal_statement"], row["id"])
        file_path = f"MiniF2FGen/{split_module}/{stem}.lean"
        file_abs = PROJECT_DIR / file_path
        file_abs.parent.mkdir(parents=True, exist_ok=True)
        statement = row["formal_statement"].strip()
        file_abs.write_text(
            (
                "import Mathlib\n\n"
                f"namespace {namespace}\n\n"
                f"{statement}\n\n"
                f"end {namespace}\n"
            ),
            encoding="utf-8",
        )
        check = subprocess.run(
            ["lake", "env", "lean", file_path],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            file_abs.unlink(missing_ok=True)
            print(f"Skipping {row['id']} due to build failure.")
            continue

        records.append(
            {
                "id": row["id"],
                "split": split,
                "file_path": file_path,
                "full_name": f"{namespace}.{theorem_name}",
                "start": [5, 1],
                "informal_stmt": row.get("informal_stmt", ""),
                "informal_proof": row.get("informal_proof", ""),
            }
        )

    return records


def _write_root_module() -> None:
    imports = []
    for split_module in ("Val", "Test"):
        split_dir = GEN_ROOT / split_module
        if not split_dir.exists():
            continue
        for lean_file in sorted(split_dir.glob("*.lean")):
            imports.append(f"import MiniF2FGen.{split_module}.{lean_file.stem}")
    (PROJECT_DIR / "MiniF2FGen.lean").write_text("\n".join(imports) + "\n", encoding="utf-8")


def _commit_project() -> str:
    safe_dir = str(PROJECT_DIR.resolve())
    git_prefix = ["git", "-c", f"safe.directory={safe_dir}"]
    if not (PROJECT_DIR / ".git").exists():
        subprocess.run([*git_prefix, "init"], cwd=PROJECT_DIR, check=True)
    subprocess.run(
        [
            *git_prefix,
            "add",
            "-A",
            "MiniF2FGen",
            "MiniF2FGen.lean",
            "lakefile.lean",
            "lean-toolchain",
            ".gitignore",
        ],
        cwd=PROJECT_DIR,
        check=True,
    )
    subprocess.run(
        [
            *git_prefix,
            "commit",
            "--allow-empty",
            "-m",
            "Generate MiniF2F project for evaluation",
        ],
        cwd=PROJECT_DIR,
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "MiniF2F Bot",
            "GIT_AUTHOR_EMAIL": "minif2f@example.com",
            "GIT_COMMITTER_NAME": "MiniF2F Bot",
            "GIT_COMMITTER_EMAIL": "minif2f@example.com",
        },
    )
    return subprocess.check_output(
        [*git_prefix, "rev-parse", "HEAD"],
        cwd=PROJECT_DIR,
        text=True,
    ).strip()


def _write_records(records: list[dict], split: str, commit: str) -> None:
    repo_url = str(PROJECT_DIR.resolve())
    payload = []
    for r in records:
        payload.append({**r, "url": repo_url, "commit": commit})
    output_path = MINIF2F_DIR / f"{split}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    shutil.rmtree(PROJECT_DIR, ignore_errors=True)
    val_rows, test_rows = _load_raw_splits()
    _write_project_files()
    subprocess.run(["lake", "update"], cwd=PROJECT_DIR, check=True)
    val_records = _write_split("val", val_rows)
    test_records = _write_split("test", test_rows)
    _write_root_module()
    commit = _commit_project()
    MINIF2F_DIR.mkdir(parents=True, exist_ok=True)
    _write_records(val_records, "val", commit)
    _write_records(test_records, "test", commit)
    print("Successfully prepared MiniF2F project and LeanDojo-style JSON files at data/minif2f/.")


if __name__ == "__main__":
    main()
