"""Script for evaluating the prover on theorems extracted by LeanDojo.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os
from datetime import datetime

os.environ["RAY_DEDUP_LOGS"] = "0"
import uuid
import json
import pickle
import hashlib
import argparse
from loguru import logger
from lean_dojo import Theorem
from typing import List, Tuple, Optional
from lean_dojo import LeanGitRepo, Theorem, Pos, get_traced_repo_path

from common import set_logger
from prover.proof_search import Status, DistributedProver


def _patch_leandojo_extractdata_tsyntax() -> None:
    """Align stock LeanDojo ``ExtractData.lean`` with Lean 4.12+: ``parseHeader`` yields ``Syntax``, but ``getImports`` expects ``TSyntax``."""
    import lean_dojo

    path = Path(lean_dojo.__file__).resolve().parent / "data_extraction" / "ExtractData.lean"
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    if "getImports (⟨header⟩ : TSyntax `Lean.Parser.Module.header)" in text:
        return
    old = "(← getImports header)"
    if old not in text:
        return
    new = "(← getImports (⟨header⟩ : TSyntax `Lean.Parser.Module.header))"
    try:
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
    except OSError as e:
        logger.warning(f"Could not patch {path} ({e}); LeanDojo tracing may fail on Lean 4.12+.")
        return
    logger.info(
        f"Patched LeanDojo ExtractData.lean for TSyntax/header typing (Lean 4.12+): {path}"
    )


def _elan_toolchain_from_local_repo(url: str) -> Optional[str]:
    """Return toolchain spec from ``lean-toolchain`` if ``url`` is a local checkout."""
    if url.startswith(("http://", "https://", "git@")):
        return None
    root = Path(url)
    if not root.is_dir():
        return None
    lt = root / "lean-toolchain"
    if not lt.is_file():
        return None
    line = lt.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    return line or None


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
    if dataset == "leandojo":
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
        else:
            assert repo_url is not None and commit is not None, (
                "repo_url and commit must be provided for custom datasets."
            )
            default_repo_url = repo_url
            default_commit = commit

        default_repo = LeanGitRepo(default_repo_url, default_commit)
        data = json.load(open(os.path.join(data_path, f"{split}.json")))
        theorems = []
        positions = []
        for t in data:
            if "file_path" not in t or "full_name" not in t:
                raise ValueError(
                    "Dataset JSON must include 'file_path' and 'full_name' for each problem. "
                    "For MiniF2F, run: python scripts/setup_minif2f_example.py"
                )
            if file_path is not None and t["file_path"] != file_path:
                continue
            if full_name is not None and t["full_name"] != full_name:
                continue
            if name_filter is not None and not hashlib.md5(
                t["full_name"].encode()
            ).hexdigest().startswith(name_filter):
                continue
            if "url" in t and "commit" in t:
                row_repo = LeanGitRepo(t["url"], t["commit"])
            else:
                row_repo = default_repo
            theorems.append(Theorem(row_repo, t["file_path"], t["full_name"]))
            if "start" in t:
                positions.append(Pos(*t["start"]))
            else:
                positions.append(Pos(1, 1))

        assert len(theorems) > 0, f"No theorems loaded from {data_path}/{split}.json"
        if num_theorems is not None:
            theorems = theorems[:num_theorems]
            positions = positions[:num_theorems]
        logger.info(f"{len(theorems)} theorems loaded from {data_path}")
        repo = theorems[0].repo

    all_repos = {thm.repo for thm in theorems}
    _patch_leandojo_extractdata_tsyntax()
    for r in all_repos:
        # LeanDojo's tracer runs bare `lean` (not `lake env lean`) for --print-prefix; elan
        # would otherwise use the user's default toolchain (often 4.30+) and break
        # ExtractData.lean against a project pinned to an older Lean (e.g. MiniF2F @ 4.29).
        tc = _elan_toolchain_from_local_repo(str(r.url))
        if tc:
            os.environ["ELAN_TOOLCHAIN"] = tc
            logger.info(f"Set ELAN_TOOLCHAIN={tc} for tracing {r}")
        # Ensures ~/.cache/lean_dojo has a trace; traces on first use (can take a while).
        get_traced_repo_path(r)

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

    metadata = json.load(open(os.path.join(data_path, "../metadata.json")))
    repo = LeanGitRepo(metadata["from_repo"]["url"], metadata["from_repo"]["commit"])

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
) -> float:
    set_logger(verbose)

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
    )
    results = prover.search_unordered(repo, theorems, positions)

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["REPROVER_LOG_FILE"] = f"logs/trace_{timestamp}.log"
    # Ensure the directory exists
    os.makedirs("logs", exist_ok=True)

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
        choices=["train", "val", "test"],
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
        choices=["leandojo", "minif2f", "veribench"],
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
    args = parser.parse_args()

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
    )

    logger.info(f"Pass@1: {pass_1}")
    logger.info(f"Configuration used: {args}")


if __name__ == "__main__":
    main()
