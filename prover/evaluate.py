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
from prover.proof_search import Status, DistributedProver

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _elan_toolchain_from_local_repo(repo_path: str) -> Optional[str]:
    path = Path(repo_path) / "lean-toolchain"
    if path.exists():
        return path.read_text().strip()
    return None


def _patch_leandojo_extract_data_compat() -> None:
    """Patch LeanDojo ExtractData for Lean parser header API differences."""
    try:
        from lean_dojo.data_extraction import trace as trace_mod
    except Exception as ex:
        logger.warning(f"Unable to import LeanDojo trace module for compatibility patch: {ex}")
        return

    extract_path = Path(inspect.getfile(trace_mod)).resolve().parent / "ExtractData.lean"
    if not extract_path.exists():
        return

    text = extract_path.read_text(encoding="utf-8")
    old = "IO.FS.writeFile dep_path (← getImports header)"
    new = "IO.FS.writeFile dep_path (← getImports ⟨header⟩)"
    if old in text:
        extract_path.write_text(text.replace(old, new), encoding="utf-8")
        logger.info(f"Patched LeanDojo ExtractData compatibility at {extract_path}")


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
        if dataset == "veribench":
            default_repo_url = repo_url or "https://github.com/shishir-h/VeriBench"
            default_commit = commit or "main"
        elif dataset == "lean_workbook":
            default_repo_url = repo_url or str((_REPO_ROOT / "data/lean_workbook_reprover/project").resolve())
            default_commit = commit or "main"
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
            theorems.append(Theorem(default_repo, t["file_path"], t["full_name"]))
            if "start" in t:
                positions.append(Pos(*t["start"]))
            else:
                positions.append(Pos(1, 1))
        repo = default_repo

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
        try:
            get_traced_repo_path(r)
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
    wall_timeout: Optional[int] = None,
) -> float:
    set_logger(verbose)
    _patch_leandojo_extract_data_compat()

    repo, theorems, positions = _get_theorems(
        data_path, split, file_path, full_name, name_filter, num_theorems, dataset, repo_url, commit
    )

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
        wall_timeout=wall_timeout,
    )
    results = prover.search_unordered(repo, theorems, positions, wall_timeout=wall_timeout)
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
        "--wall-timeout",
        type=int,
        default=None,
        help="Maximum number of seconds the entire evaluation can take.",
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
        args.wall_timeout,
    )

    logger.info(f"Pass@1: {pass_1}")
    logger.info(f"Configuration used: {args}")


if __name__ == "__main__":
    main()
