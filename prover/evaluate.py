"""Script for evaluating the prover on theorems extracted by LeanDojo.
"""

import os
from datetime import datetime
import subprocess
import inspect

os.environ["RAY_DEDUP_LOGS"] = "0"
import uuid
import json
import pickle
import hashlib
import argparse
from pathlib import Path
from loguru import logger
from lean_dojo import Theorem
from typing import List, Tuple, Optional
from lean_dojo import LeanGitRepo, Theorem, Pos, is_available_in_cache
from lean_dojo.data_extraction.trace import get_traced_repo_path

from common import set_logger
from prover.attempt_summary import AttemptRecord, write_jsonl
from prover.proof_search import Status, DistributedProver

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _elan_toolchain_from_local_repo(repo_path: str) -> Optional[str]:
    path = Path(repo_path) / "lean-toolchain"
    if path.exists():
        return path.read_text().strip()
    return None


def _patch_leandojo_check_files(trace_mod) -> None:
    """Make LeanDojo's ``check_files`` count oleans using the actual olean layout.

    LeanDojo's ``check_files`` globs ``**/build/lib/lean/**/*.olean``, but Lean
    (<= v4.12) lays oleans out under ``.lake/build/lib/`` with no ``lean/``
    segment, so the original glob matches nothing and ``len(jsons) <= len(oleans)``
    fails. We re-implement it against ``build/lib/`` and reassign the module-level
    name so ``_trace`` picks it up.
    """
    from pathlib import Path as _Path

    def check_files(packages_path, no_deps: bool) -> None:
        cwd = _Path.cwd()
        packages_path = cwd / packages_path
        jsons = {
            p.with_suffix("").with_suffix("")
            for p in cwd.glob("**/build/ir/**/*.ast.json")
            if not no_deps or not p.is_relative_to(packages_path)
        }
        deps = {
            p.with_suffix("")
            for p in cwd.glob("**/build/ir/**/*.dep_paths")
            if not no_deps or not p.is_relative_to(packages_path)
        }
        oleans = {
            _Path(str(p.with_suffix("")).replace("/build/lib/", "/build/ir/"))
            for p in cwd.glob("**/build/lib/**/*.olean")
            if not no_deps or not p.is_relative_to(packages_path)
        }
        assert len(jsons) <= len(oleans) and len(deps) <= len(oleans)
        missing_jsons = {p.with_suffix(".ast.json") for p in oleans - jsons}
        missing_deps = {p.with_suffix(".dep_paths") for p in oleans - deps}
        if len(missing_jsons) > 0 or len(missing_deps) > 0:
            for p in missing_jsons.union(missing_deps):
                trace_mod.logger.warning(f"Missing {p}")

    trace_mod.check_files = check_files
    logger.info("Patched LeanDojo check_files for the .lake/build/lib olean layout.")


def _patch_leandojo_extract_data_compat() -> None:
    """Patch LeanDojo ExtractData/trace for Lean parser API and olean-layout differences."""
    try:
        from lean_dojo.data_extraction import trace as trace_mod
    except Exception as ex:
        logger.warning(f"Unable to import LeanDojo trace module for compatibility patch: {ex}")
        return

    extract_path = Path(inspect.getfile(trace_mod)).resolve().parent / "ExtractData.lean"
    if extract_path.exists():
        text = extract_path.read_text(encoding="utf-8")
        original = text
        # Lean parser header API: getImports expects a TSyntax header.
        text = text.replace(
            "IO.FS.writeFile dep_path (← getImports header)",
            "IO.FS.writeFile dep_path (← getImports ⟨header⟩)",
        )
        # Olean layout: Lean (<= v4.12) places oleans under `.lake/build/lib/`
        # (no `lean/` segment). findLean strips `build/lib/lean/`, which leaves a
        # nonexistent source path and panics at `assert! path.pathExists`. Strip
        # `build/lib/` instead so olean -> source resolution works.
        text = text.replace(
            '.replace ".lake/build/lib/lean/" ""',
            '.replace ".lake/build/lib/" ""',
        ).replace(
            '|>.replace "build/lib/lean/" ""',
            '|>.replace "build/lib/" ""',
        )
        # HashMap API: Lean v4.11's `Lean.HashMap` uses `.find?` for lookup, but
        # newer LeanDojo's ExtractData calls `.get?` (the Std.HashMap name), which
        # doesn't exist on `env.const2ModIdx` (type `HashMap Name ModuleIdx`).
        text = text.replace(
            "env.const2ModIdx.get? fullName",
            "env.const2ModIdx.find? fullName",
        )
        # Same olean-layout mismatch as above, but in `shouldProcess`: it builds
        # the expected olean path under `lib/lean` to decide whether a file has
        # been built and should be traced. Since real oleans live under `lib`
        # (no `lean/` segment) for Lean <= v4.12, every path check misses,
        # `shouldProcess` returns false for every file, and extraction silently
        # processes nothing.
        text = text.replace(
            'Path.toBuildDir "lib/lean" relativePath "olean"',
            'Path.toBuildDir "lib" relativePath "olean"',
        )
        if text != original:
            extract_path.write_text(text, encoding="utf-8")
            logger.info(f"Patched LeanDojo ExtractData compatibility at {extract_path}")

    _patch_leandojo_check_files(trace_mod)


def _is_preprocessed_theorem_record(record: dict) -> bool:
    required_keys = {"file_path", "full_name", "start", "url", "commit"}
    return required_keys.issubset(record.keys())


def _resolve_minif2f_data_path(data_path: str, split: str) -> str:
    split_file = Path(data_path) / f"{split}.json"
    if split_file.exists():
        data = json.loads(split_file.read_text())
        if len(data) > 0 and _is_preprocessed_theorem_record(data[0]):
            return data_path
        if len(data) > 0:
            fallback = Path(data_path) / "minif2f"
            fallback_split = fallback / f"{split}.json"
            if fallback_split.exists():
                logger.warning(
                    f"Detected raw MiniF2F rows in {split_file}; using {fallback} instead."
                )
                return str(fallback)
    fallback = Path(data_path) / "minif2f"
    fallback_split = fallback / f"{split}.json"
    if fallback_split.exists():
        return str(fallback)
    raise ValueError(
        "MiniF2F records are not in LeanDojo format. Run "
        "`python scripts/setup_minif2f_example.py` and use "
        "`--data-path data/minif2f`."
    )


def _get_theorems(
    data_path: str,
    split: str,
    file_path: str,
    full_name: str,
    name_filter: str,
    num_theorems: int,
    dataset: str = "leandojo",
    repo_url: Optional[str] = None,
    commit: Optional[str] = None,
) -> Tuple[LeanGitRepo, List[Theorem], List[Pos]]:
    if dataset == "minif2f":
        data_path = _resolve_minif2f_data_path(data_path, split)

    if dataset in {"leandojo", "minif2f"}:
        repo, theorems, positions = _get_theorems_from_files(
            data_path,
            split,
            file_path,
            full_name,
            name_filter,
            num_theorems,
        )
    else:
        if dataset == "minif2f":
            default_repo_url = repo_url or "https://github.com/leanprover-community/mathlib4"
            default_commit = commit or "master"
        elif dataset == "veribench":
            default_repo_url = repo_url or "https://github.com/shishir-h/VeriBench"
            default_commit = commit or "main"
        elif dataset == "lean_workbook":
            default_repo_url = repo_url or str((_REPO_ROOT / "data/lean_workbook_reprover/project").resolve())
            try:
                head_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=default_repo_url, text=True
                ).strip()
            except Exception:
                head_commit = "master"
            default_commit = commit or head_commit
        else:
            assert repo_url is not None and commit is not None, (
                "repo_url and commit must be provided for custom datasets."
            )
            default_repo_url = repo_url
            default_commit = commit

        import shutil

        shutil.rmtree("project", ignore_errors=True)
        default_repo = LeanGitRepo(default_repo_url, default_commit)
        data = json.load(open(os.path.join(data_path, f"{split}.json")))
        theorems = []
        positions = []
        for t in data:
            if file_path is not None and t["file_path"] != file_path:
                continue
            if full_name is not None and t["full_name"] != full_name:
                continue
            if name_filter is not None and not hashlib.md5(
                t["full_name"].encode()
            ).hexdigest().startswith(name_filter):
                continue
            theorems.append(Theorem(default_repo, t["file_path"], t["full_name"]))
            if "start" in t:
                positions.append(Pos(*t["start"]))
            else:
                positions.append(Pos(1, 1))
        repo = default_repo

        # Deterministically order and cap to num_theorems, mirroring
        # _get_theorems_from_files so --num-theorems / --name-filter work for
        # custom datasets (lean_workbook, veribench, ...) too.
        if len(theorems) > 0:
            theorems_and_positions = list(zip(theorems, positions))
            theorems_and_positions.sort(
                key=lambda x: hashlib.md5(
                    f"{x[0].file_path}:{x[0].full_name}".encode()
                ).hexdigest()
            )
            theorems, positions = map(list, zip(*theorems_and_positions))
        if num_theorems is not None:
            theorems = theorems[:num_theorems]
            positions = positions[:num_theorems]
        logger.info(f"{len(theorems)} theorems loaded from {data_path}")

    all_repos = {thm.repo for thm in theorems}
    for r in all_repos:
        # LeanDojo's tracer runs bare `lean` (not `lake env lean`) for --print-prefix; elan
        # would otherwise use the user's default toolchain (often 4.30+) and break
        # ExtractData.lean against a project pinned to an older Lean (e.g. MiniF2F @ 4.29).
        tc = _elan_toolchain_from_local_repo(str(r.url))
        if tc:
            os.environ["ELAN_TOOLCHAIN"] = tc
            logger.info(f"Set ELAN_TOOLCHAIN={tc} for tracing {r}")
        # Ensures ~/.cache/lean_dojo has a trace; traces on first use (can take a while).
        # build_deps=False downloads prebuilt deps (`lake exe cache get`) and only
        # AST-extracts this repo's own files (not all of mathlib) -- much faster.
        # Tradeoff: no dependency premises, so a premise retriever has no corpus.
        try:
            get_traced_repo_path(r, build_deps=False)
        except subprocess.CalledProcessError:
            logger.warning(f"Tracing {r} failed (lake build exited non-zero). Continuing anyway, as some OLEANs may have been built.")

    return repo, theorems, positions


def _get_theorems_from_files(
    data_path: str,
    split: str,
    file_path: Optional[str],
    full_name: Optional[str],
    name_filter: Optional[str],
    num_theorems: Optional[int],
) -> Tuple[LeanGitRepo, List[Theorem], List[Pos]]:
    data = json.load(open(os.path.join(data_path, f"{split}.json")))
    if len(data) == 0:
        raise ValueError(f"No theorems found in {os.path.join(data_path, f'{split}.json')}")
    if not _is_preprocessed_theorem_record(data[0]):
        raise ValueError(
            f"{os.path.join(data_path, f'{split}.json')} is missing one or more required keys: "
            "file_path, full_name, start, url, commit."
        )
    theorems = []
    positions = []

    import shutil
    shutil.rmtree("project", ignore_errors=True)
    for t in data:
        if file_path is not None and t["file_path"] != file_path:
            continue
        if full_name is not None and t["full_name"] != full_name:
            continue
        if name_filter is not None and not hashlib.md5(
            t["full_name"].encode()
        ).hexdigest().startswith(name_filter):
            continue
        repo = LeanGitRepo(t["url"], t["commit"])
        theorems.append(Theorem(repo, t["file_path"], t["full_name"]))
        positions.append(Pos(*t["start"]))

    # Jointly sort theorems and positions
    assert len(theorems) > 0
    theorems_and_positions = list(zip(theorems, positions))
    theorems_and_positions.sort(
        key=lambda x: hashlib.md5(
            f"{x[0].file_path}:{x[0].full_name}".encode()
        ).hexdigest()
    )
    theorems, positions = zip(*theorems_and_positions)
    theorems, positions = list(theorems), list(positions)

    if num_theorems is not None:
        theorems = theorems[:num_theorems]
        positions = positions[:num_theorems]
    logger.info(f"{len(theorems)} theorems loaded from {data_path}")

    metadata_path = os.path.join(data_path, "../metadata.json")
    if os.path.exists(metadata_path):
        metadata = json.load(open(metadata_path))
        repo = LeanGitRepo(metadata["from_repo"]["url"], metadata["from_repo"]["commit"])
    else:
        repo = theorems[0].repo

    return repo, theorems, positions


def evaluate(
    data_path: str,
    exp_id: Optional[str] = None,
    split: str = "val",
    file_path: Optional[str] = None,
    full_name: Optional[str] = None,
    name_filter: Optional[str] = None,
    num_theorems: Optional[int] = None,
    use_vllm: bool = False,
    gen_ckpt_path: Optional[str] = None,
    ret_ckpt_path: Optional[str] = None,
    indexed_corpus_path: Optional[str] = None,
    max_inp_seq_len: int = 2048,
    max_oup_seq_len: int = 512,
    length_penalty: float = 0.0,
    tactic: Optional[str] = None,
    module: Optional[str] = None,
    num_sampled_tactics: int = 64,
    timeout: int = 600,
    max_expansions: Optional[int] = None,
    num_workers: int = 1,
    num_gpus: int = 0,
    save_results: bool = False,
    verbose: bool = False,
    algorithm: str = "best",
    dataset: str = "leandojo",
    repo_url: Optional[str] = None,
    commit: Optional[str] = None,
    repair_ckpt_path: Optional[str] = None,
    repair_count: int = 1,
    timeout_accounting: str = "wall",
    global_wall_timeout: Optional[int] = None,
    summary_jsonl: Optional[str] = None,
    resume: bool = False,
) -> float:
    args_log_file = os.getenv("REPROVER_LOG_FILE")
    if resume and args_log_file and os.path.exists(args_log_file):
        os.environ["REPROVER_LOG_MODE"] = "a"

    set_logger(verbose)
    _patch_leandojo_extract_data_compat()

    repo, theorems, positions = _get_theorems(
        data_path, split, file_path, full_name, name_filter, num_theorems, dataset, repo_url, commit
    )

    previous_results = []
    if resume and args_log_file and os.path.exists(args_log_file):
        logger.info(f"Resuming from existing log: {args_log_file}")
        import re
        import ast
        from prover.proof_search import SearchResult
        from lean_dojo import Theorem

        sr_pattern = re.compile(
            r"SearchResult\(theorem=Theorem\(.*?, file_path=PosixPath\('(?P<file_path>[^']+)'\), full_name='(?P<full_name>[^']+)'\), "
            r"status=<Status\.(?P<status>\w+): '[^']+'\>, proof=(?P<proof>None|\[.*?\]), "
            r"actor_time=(?P<actor_time>[0-9.eE+-]+), "
            r"environment_time=(?P<environment_time>[0-9.eE+-]+), "
            r"repair_time=(?P<repair_time>[0-9.eE+-]+), "
            r"total_time=(?P<total_time>[0-9.eE+-]+), "
            r"num_total_nodes=(?P<num_total_nodes>\d+), "
            r"num_searched_nodes=(?P<num_searched_nodes>\d+)\)"
        )

        completed_thms = {}
        with open(args_log_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = sr_pattern.search(line)
                if m:
                    status_str = m.group("status")
                    if status_str == "PROVED":
                        status = Status.PROVED
                    elif status_str == "FAILED":
                        status = Status.FAILED
                    else:
                        status = Status.OPEN

                    proof_str = m.group("proof")
                    proof = None
                    if proof_str != "None":
                        try:
                            proof = ast.literal_eval(proof_str)
                        except Exception:
                            proof = []

                    thm_name = m.group("full_name")
                    completed_thms[thm_name] = SearchResult(
                        theorem=Theorem(repo, Path(m.group("file_path")), thm_name),
                        status=status,
                        proof=proof,
                        actor_time=float(m.group("actor_time")),
                        environment_time=float(m.group("environment_time")),
                        repair_time=float(m.group("repair_time")),
                        total_time=float(m.group("total_time")),
                        num_total_nodes=int(m.group("num_total_nodes")),
                        num_searched_nodes=int(m.group("num_searched_nodes"))
                    )

        logger.info(f"Found {len(completed_thms)} completed theorems in log.")

        new_theorems = []
        new_positions = []
        for thm, pos in zip(theorems, positions):
            if thm.full_name not in completed_thms:
                new_theorems.append(thm)
                new_positions.append(pos)
            else:
                previous_results.append(completed_thms[thm.full_name])

        logger.info(f"Filtered theorems: {len(theorems)} -> {len(new_theorems)} remaining to run.")
        theorems = new_theorems
        positions = new_positions

    # Search for proofs using multiple concurrent provers.
    prover = DistributedProver(
        use_vllm,
        gen_ckpt_path,
        ret_ckpt_path,
        indexed_corpus_path,
        max_inp_seq_len,
        max_oup_seq_len,
        length_penalty,
        tactic,
        module,
        num_workers,
        num_gpus=num_gpus,
        timeout=timeout,
        max_expansions=max_expansions,
        num_sampled_tactics=num_sampled_tactics,
        debug=verbose,
        algorithm=algorithm,
        repair_ckpt_path=repair_ckpt_path,
        repair_count=repair_count,
        timeout_accounting=timeout_accounting,
    )
    if len(theorems) > 0:
        new_results = prover.search_unordered(repo, theorems, positions, global_wall_timeout=global_wall_timeout)
    else:
        new_results = []
    results = previous_results + new_results
    # Calculate the result statistics.
    num_proved = num_failed = num_discarded = 0
    for r in results:
        if r is None:
            num_discarded += 1
        elif r.status == Status.PROVED:
            num_proved += 1
        else:
            num_failed += 1

    logger.info(
        f"Evaluation done! {num_proved} theorems proved, {num_failed} theorems failed, {num_discarded} non-theorems discarded"
    )

    if num_proved + num_failed == 0:
        pass_1 = float("nan")
    else:
        pass_1 = num_proved / (num_proved + num_failed)

    # Save the results.
    if exp_id is None:
        exp_id = str(uuid.uuid4())
    if save_results:
        pickle_path = f"{exp_id}_results.pickle"
        pickle.dump(results, open(pickle_path, "wb"))
        logger.info(f"Results saved to {pickle_path}")

    if summary_jsonl:
        records = []
        for r in results:
            if r is None:
                continue
            records.append(
                AttemptRecord(
                    theorem=r.theorem.full_name,
                    status=r.status.value,
                    total_time=r.total_time,
                    repair_time=r.repair_time,
                    actor_time=r.actor_time,
                    environment_time=r.environment_time,
                    num_total_nodes=r.num_total_nodes,
                    num_searched_nodes=r.num_searched_nodes,
                )
            )
        write_jsonl(summary_jsonl, records)
        logger.info(f"Attempt summary saved to {summary_jsonl}")

    return pass_1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script for evaluating the prover on theorems extracted by LeanDojo."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the data extracted by LeanDojo (e.g., data/leandojo_benchmark/random).",
    )
    parser.add_argument("--exp-id", type=str, help="Experiment ID used for logging.")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "lw_val", "lw_test"],
        default="val",
    )
    # `file_path`, `full_name`, `name_filter`, and `num_theorems` can be used to filter theorems.
    parser.add_argument("--file-path", type=str)
    parser.add_argument("--full-name", type=str)
    parser.add_argument("--name-filter", type=str)
    parser.add_argument("--num-theorems", type=int)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument(
        "--gen_ckpt_path",
        type=str,
        help="Checkpoint of the tactic generator.",
    )
    parser.add_argument(
        "--ret_ckpt_path",
        type=str,
        help="Checkpoint of the premise retriever.",
    )
    parser.add_argument(
        "--indexed-corpus-path",
        type=str,
        help="Path to a pickled indexed corpus. Not required for models w/o retrieval.",
    )
    parser.add_argument("--max-inp-seq-len", type=int, default=2048)
    parser.add_argument("--max-oup-seq-len", type=int, default=512)
    parser.add_argument("--length-penalty", type=float, default=0.0)
    parser.add_argument("--tactic", type=str, help="The tactic to evaluate.")
    parser.add_argument("--module", type=str, help="The module to import the tactic.")
    parser.add_argument(
        "--num-sampled-tactics",
        type=int,
        default=64,
        help="Number of tactics to sample at each node during proof search.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Maximum number of seconds the proof search can take.",
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=None,
        help="Maximum number of expansions during proof search.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="The number of concurrent provers."
    )
    parser.add_argument(
        "--num-gpus", type=int, default=0, help="The number of GPUs for proof search."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["best", "bfs", "dfs"],
        default="best",
        help="The search algorithm to use.",
    )
    parser.add_argument(
        "--repair-ckpt-path",
        type=str,
        help="Checkpoint of the error repair model.",
    )
    parser.add_argument(
        "--repair-count",
        type=int,
        default=1,
        help="Number of repair attempts to try if a tactic fails.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["leandojo", "minif2f", "veribench", "lean_workbook"],
        default="leandojo",
        help="The dataset to evaluate on.",
    )
    parser.add_argument(
        "--repo-url",
        type=str,
        default=None,
        help="The URL of the repository (required for custom datasets).",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help="The commit hash (required for custom datasets).",
    )
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument(
        "--verbose", action="store_true", help="Set the logging level to DEBUG."
    )
    parser.add_argument(
        "--timeout-accounting",
        choices=["wall", "effective"],
        default="wall",
        help=(
            "How to charge local per-theorem timeout. "
            "'wall' charges all elapsed time; 'effective' preserves the legacy "
            "best-first behavior that subtracts repair overhead."
        ),
    )
    parser.add_argument(
        "--global-wall-timeout",
        "--wall-timeout",
        dest="global_wall_timeout",
        type=int,
        default=None,
        help=(
            "Maximum number of wall-clock seconds for the entire evaluation run. "
            "--wall-timeout is kept as a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--summary-jsonl",
        type=str,
        default=None,
        help="Optional path for structured per-theorem attempt summaries.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume evaluation from the existing log file by skipping already attempted theorems.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to the log file.",
    )
    args = parser.parse_args()

    # Set up logging
    if args.log_file:
        os.environ["REPROVER_LOG_FILE"] = args.log_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.environ["REPROVER_LOG_FILE"] = f"logs/trace_{timestamp}.log"
    os.makedirs("logs", exist_ok=True)

    assert args.gen_ckpt_path or args.tactic
    assert args.num_gpus <= args.num_workers

    logger.info(f"PID: {os.getpid()}")
    logger.info(args)

    pass_1 = evaluate(
        args.data_path,
        args.exp_id,
        args.split,
        args.file_path,
        args.full_name,
        args.name_filter,
        args.num_theorems,
        args.use_vllm,
        args.gen_ckpt_path,
        args.ret_ckpt_path,
        args.indexed_corpus_path,
        args.max_inp_seq_len,
        args.max_oup_seq_len,
        args.length_penalty,
        args.tactic,
        args.module,
        args.num_sampled_tactics,
        args.timeout,
        args.max_expansions,
        args.num_workers,
        args.num_gpus,
        args.save_results,
        args.verbose,
        args.algorithm,
        args.dataset,
        args.repo_url,
        args.commit,
        args.repair_ckpt_path,
        args.repair_count,
        args.timeout_accounting,
        args.global_wall_timeout,
        args.summary_jsonl,
        args.resume,
    )

    logger.info(f"Pass@1: {pass_1}")
    logger.info(f"Configuration used: {args}")


if __name__ == "__main__":
    main()
