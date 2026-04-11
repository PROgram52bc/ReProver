import json
import argparse
from glob import glob
from loguru import logger
from lean_dojo import LeanGitRepo, trace, is_available_in_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data")
    args = parser.parse_args()
    logger.info(args)

    json_paths = set(glob(f"{args.data_path}/*.json"))
    json_paths.update(glob(f"{args.data_path}/*/*/*.json"))

    url_commits = set()
    for path in sorted(json_paths):
        data = json.load(open(path))
        if not isinstance(data, list):
            continue
        for ex in data:
            if not isinstance(ex, dict) or "url" not in ex or "commit" not in ex:
                continue
            url_commits.add((ex["url"], ex["commit"]))

    repos = set()
    for url, commit in url_commits:
        repo = LeanGitRepo(url, commit)
        if not is_available_in_cache(repo) and not repo.is_lean4:
            repos.add(repo)

    logger.info(f"Repos to trace: {repos}")

    for repo in repos:
        trace(repo)


if __name__ == "__main__":
    main()
