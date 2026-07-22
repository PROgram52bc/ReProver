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
# Pinned to v4.11.0 to match the working lean_workbook setup: LeanDojo 4.20.0's
# Lean4Repl stdin read loop is broken under Lean v4.12.0 (the REPL initializes
# and prints the goal, but `IO.getStdin.getLine` returns empty for every tactic,
# so the process exits 1 on the first tactic). v4.11.0 reads stdin correctly.
LEAN_TOOLCHAIN = "leanprover/lean4:v4.11.0"
MATHLIB_COMMIT = "v4.11.0"


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
    # Parse the actual declaration name (the source of truth LeanDojo locates by).
    # NOTE: must be a single-backslash `\s`/`\b` regex -- `\\s` in a raw string is a
    # literal backslash and never matches, silently falling back to the id.
    match = re.search(r"(?m)^\s*(?:theorem|lemma)\s+([A-Za-z0-9_'.]+)", formal_statement)
    return match.group(1) if match else fallback.replace("-", "_")


def _write_split_files(split: str, rows: list[dict]) -> list[dict]:
    """Write every candidate `.lean` file for a split and return its metadata.

    Validation is deferred: instead of starting one Lean process per theorem
    (each re-importing all of Mathlib), we write everything and validate with a
    single parallel `lake build` in `main`.
    """
    split_module = "Val" if split == "val" else "Test"
    split_dir = GEN_ROOT / split_module
    split_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []

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
        entries.append(
            {
                "row": row,
                "split": split,
                "split_module": split_module,
                "stem": stem,
                "namespace": namespace,
                "theorem_name": theorem_name,
                "file_path": file_path,
                "file_abs": file_abs,
            }
        )

    return entries


def _olean_path(entry: dict) -> Path:
    # Lean <= v4.12 lays oleans out under `.lake/build/lib/` (no `lean/` segment).
    return (
        PROJECT_DIR
        / ".lake"
        / "build"
        / "lib"
        / "MiniF2FGen"
        / entry["split_module"]
        / f"{entry['stem']}.olean"
    )


def _collect_records(entries: list[dict]) -> list[dict]:
    """Keep entries whose olean was produced by the batched build; drop the rest."""
    records: list[dict] = []
    for e in entries:
        if not _olean_path(e).exists():
            e["file_abs"].unlink(missing_ok=True)
            print(f"Skipping {e['row']['id']} due to build failure.")
            continue
        records.append(
            {
                "id": e["row"]["id"],
                "split": e["split"],
                "file_path": e["file_path"],
                "full_name": f"{e['namespace']}.{e['theorem_name']}",
                "start": [5, 1],
                "informal_stmt": e["row"].get("informal_stmt", ""),
                "informal_proof": e["row"].get("informal_proof", ""),
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
    # Fetch prebuilt Mathlib oleans so per-file `lake env lean` validation can
    # resolve `import Mathlib` (otherwise every theorem fails to typecheck and is
    # skipped, producing empty val/test JSON).
    subprocess.run(["lake", "exe", "cache", "get"], cwd=PROJECT_DIR, check=True)

    # Write every candidate file, then validate them all with one parallel
    # `lake build` (independent modules build concurrently; a failing theorem
    # only fails its own module). The first build's exit code is ignored because
    # some candidates are expected to fail.
    val_entries = _write_split_files("val", val_rows)
    test_entries = _write_split_files("test", test_rows)
    _write_root_module()
    print("Building all candidate theorems (single parallel lake build)...")
    subprocess.run(["lake", "build"], cwd=PROJECT_DIR)

    val_records = _collect_records(val_entries)
    test_records = _collect_records(test_entries)

    # Rewrite the root to import only the theorems that built, then do a final
    # clean build so the committed project compiles with a zero exit code
    # (LeanDojo requires `lake build` to succeed when tracing).
    _write_root_module()
    subprocess.run(["lake", "build"], cwd=PROJECT_DIR, check=True)

    commit = _commit_project()
    MINIF2F_DIR.mkdir(parents=True, exist_ok=True)
    _write_records(val_records, "val", commit)
    _write_records(test_records, "test", commit)
    print("Successfully prepared MiniF2F project and LeanDojo-style JSON files at data/minif2f/.")


if __name__ == "__main__":
    main()
