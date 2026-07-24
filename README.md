# ReProver Baseline Setup Guide

## 1. System Prerequisites

You must install `elan` (Lean Version Manager) before setting up the Python environment.

```bash
# 1. Install elan
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
source $HOME/.elan/env

# 2. Install system tools (Ubuntu/Debian)
sudo apt-get install git curl wget
```

## 2. Python Environment

Set up the environment with Python 3.11 and the specific dependencies required by ReProver.

```bash
# 1. Create Conda environment
conda create --yes --name ReProver python=3.11 ipython
conda activate ReProver

# 2. Install PyTorch
pip install torch

# 3. Install ReProver dependencies
pip install tqdm loguru deepspeed "pytorch-lightning[extra]" transformers wandb openai rank_bm25 git+https://github.com/PROgram52bc/LeanDojo.git vllm datasets huggingface_hub
```

## 3. Configuration

LeanDojo requires a GitHub Access Token to trace repositories and map definitions.

1.  Generate a token [here](https://github.com/settings/tokens) (Classic, `public_repo` scope).
2.  Export it in your shell by creating a copy of `env.sh.template`:

```bash
cp env.sh.template env.sh
echo 'export GITHUB_ACCESS_TOKEN="your_token_starting_with_ghp_..."' >> env.sh
```

And every time you log back in, you just need to

```bash
source env.sh
```

## 4. Data Setup & Verification

Clone the repo and verify the installation.

```bash
git clone https://github.com/PROgram52bc/ReProver.git
cd ReProver
```

On the first time, you'll need to download the leandojo benchmark by running

```bash
python scripts/download_data.py
```

### Supported Datasets

- **LeanDojo Benchmark 4**: Standard dataset (default).
- **MiniF2F**: Mathematical olympiad problems. See below for setup instructions.
- **Lean Workbook**: Contest-level math problems. See below for setup instructions
- **VeriBench**: Formal verification benchmarks.

### MiniF2F Dataset Setup

MiniF2F must be preprocessed into LeanDojo-style theorem records before evaluation.

1. **Download and Transform**:

   ```bash
   python scripts/setup_minif2f_example.py
   ```

   This script downloads (or reuses) MiniF2F rows, generates a local Lean project at `data/minif2f/project`, and writes LeanDojo-style theorem files to `data/minif2f/val.json` and `data/minif2f/test.json`.

2. **Evaluate**:
   ```bash
   python prover/evaluate.py \
       --data-path data/minif2f \
       --dataset minif2f \
       --gen_ckpt_path kaiyuy/leandojo-lean4-tacgen-byt5-small \
       --num-sampled-tactics 5 \
       --num-theorems 50
   ```

### Lean Workbook Dataset Setup

Like MiniF2F, Lean Workbook must be preprocessed into LeanDojo-style theorem records before evaluation.

1. **Download and Transform**:

   ```bash
   python scripts/setup_lean_workbook.py
   ```

   This script downloads the `InternLM/Lean-Workbook` dataset (first 100 rows as `val`, next 100 as `test`), generates a local Lean project at `data/lean_workbook_reprover/project`, fetches a prebuilt Mathlib cache so it doesn't compile Mathlib from source, verifies each theorem with an individual `lake build` (skipping any that fail to compile), and writes LeanDojo-style theorem files to `data/lean_workbook_reprover/val.json` and `data/lean_workbook_reprover/test.json`.

2. **Evaluate**:
   ```bash
   python prover/evaluate.py \
       --data-path data/lean_workbook_reprover \
       --dataset lean_workbook \
       --split val \
       --gen_ckpt_path kaiyuy/leandojo-lean4-tacgen-byt5-small \
       --num-sampled-tactics 5 \
       --num-theorems 50
   ```

   The `lean_workbook` dataset resolves its repository URL and commit automatically from the generated project, so `--repo-url` and `--commit` are not required. Use `--split test` to evaluate on the test split instead.

   > **Hitting a cache-download failure, `Pass@1: nan`, or a setup/extraction step that looks hung?** See [`LEAN_WORKBOOK_SETUP_NOTES.md`](LEAN_WORKBOOK_SETUP_NOTES.md) for three environment-specific issues (a stale-cached-curl bug, a LeanDojo olean-path bug, and a filesystem-contention issue) and their fixes, none of which are specific to this dataset's code path but which reliably show up on shared HPC filesystems.

## 5. Running the Baseline (Reproduction)

To reproduce the Pass@1 metric on LeanDojo (default):

```bash
python prover/evaluate.py \
	--data-path data/leandojo_benchmark_4/random \
	--gen_ckpt_path kaiyuy/leandojo-lean4-tacgen-byt5-small \
	--num-sampled-tactics 5 \
	--num-theorems 50
```

## 6. Retrieval-Augmented Generation (RAG)

To use retrieval-augmented generation, first index the corpus:

```bash
python retrieval/index.py \
    --ckpt_path kaiyuy/leandojo-lean4-retriever-byt5-small \
    --corpus-path data/leandojo_benchmark_4/corpus.jsonl \
    --output-path data/leandojo_benchmark_4/indexed_corpus.pkl \
    --batch-size 16
```

Then run evaluation with retrieval:

```bash
python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --gen_ckpt_path kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small \
    --ret_ckpt_path kaiyuy/leandojo-lean4-retriever-byt5-small \
    --indexed-corpus-path data/leandojo_benchmark_4/indexed_corpus.pkl \
    --num-sampled-tactics 5 \
    --num-theorems 50
```

## 7. Proof Repair

To use proof repair during search, specify a repair model checkpoint. You can also specify the number of recursive repair attempts with `--repair-count`.

### Using the gAPRIL Model (Hugging Face)

````bash
python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random \
    --gen_ckpt_path kaiyuy/leandojo-lean4-tacgen-byt5-small \
    --repair-ckpt-path uw-math-ai/gAPRIL-wo-exp \
    --repair-count 2 \
    --num-sampled-tactics 5 \
    --num-theorems 50

### Timeout Accounting for Repair Experiments

Use `--timeout` for the local per-theorem search budget. Use
`--timeout-accounting wall` to charge all elapsed time, including repair model
calls, against that local budget. Use `--timeout-accounting effective` only to
reproduce the legacy best-first behavior, where measured repair overhead is
subtracted from the local budget.

Use `--global-wall-timeout` for a whole-run wall-clock cutoff across the theorem
list. The older `--wall-timeout` spelling is accepted as an alias for
`--global-wall-timeout`.

```bash
python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random \
    --gen_ckpt_path kaiyuy/leandojo-lean4-tacgen-byt5-small \
    --timeout 600 \
    --timeout-accounting wall \
    --global-wall-timeout 100000 \
    --num-theorems 100
````

````

## 8. Training with APRIL Dataset
The APRIL dataset contains 260K Lean proof-repair tuples.

### Fast Start (Automated Script)
```bash
chmod +x scripts/train_april.sh
./scripts/train_april.sh
````

### Manual Steps

1. **Download the APRIL data**:
   ```bash
   python scripts/download_april.py
   ```
2. **Train the model**:
   ```bash
   python -m generation.train_april_model \
       --data-dir data/april \
       --model-name google/byt5-small \
       --batch-size 4 \
       --epochs 5 \
       --log-dir logs/april_repair_train
   ```

## 9. Analysis & Visualization

For qualitative analysis, search path visualization, and performance profiling, please see the [Search Graph Guide](search_graph.md).

This guide covers:

- Identifying gains and regressions between runs using `compare_outcomes.py`.
- Visualizing search trees and repair chains using `plot_search_tree.py`.
- Analyzing runtime distribution with `plot_profile.py`.

## 10. Summary of Results

| Algorithm | Configuration      | Num Tactics | Num Theorems | Pass@1       |
| --------- | ------------------ | ----------- | ------------ | ------------ |
| BEST      | BEST-Non-retrieval | 5           | 50           | 0.24         |
| BEST      | BEST-Retrieval     | 5           | 50           | 0.34         |
| DFS       | DFS-retrieval      | 5           | 50           | 0.34         |
| BFS       | BFS-retrieval      | 5           | 50           | 0.34         |
| BEST      | BEST-Non-retrieval | 64          | 200          | 0.3990       |
| BEST      | BEST-Non-retrieval | 128         | 200          | 0.4040404040 |
| BEST      | BEST-Retrieval     | 64          | 200          | 0.4394       |
| BEST      | BEST-Non-retrieval | 5           | 200          | 0.2727       |
| BEST      | BEST-Retrieval     | 5           | 200          | 0.3384       |
| BEST      | gAPRIL-Repair      | 5           | 50           | 0.22         |

## 11. Logging & Profiling

Detailed traces are saved to `logs/trace_<YYYYMMDD_HHMMSS>.log`.

### Log Content

- **`[TREE_NODE]` and `[TREE_EDGE]`**: Structural information for plotting the search tree.
- **`[PROFILE]`**: Runtime events (Generation, Lean interaction, Repair) for performance analysis.
- **Compiler Feedback**: The exact response from the Lean compiler for every tactic attempt.
- **Configuration**: The full set of CLI arguments used for the run is logged at the end.

### Proving a Single Theorem

You can run proof search for a specific theorem using the `--full-name` argument:

```bash
python prover/evaluate.py --full-name "Nat.add_comm" ...
```
