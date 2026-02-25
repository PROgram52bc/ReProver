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
# might need `module load conda` on scholar, or install miniconda
conda create --yes --name ReProver python=3.11 ipython
conda activate ReProver

# 2. Install PyTorch
# Note: Check https://pytorch.org/ for your specific CUDA version command if not using default.
pip install torch

# 3. Install ReProver dependencies
pip install tqdm loguru deepspeed "pytorch-lightning[extra]" transformers wandb openai rank_bm25 git+https://github.com/PROgram52bc/LeanDojo.git vllm
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

> Note: `env.sh.template` also does other settings useful for the project, so make sure to source it.

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

*TODO: Add instruction to manually place the MiniF2F data in the `data/` directory.*

## 5. Running the Baseline (Reproduction)
To reproduce the Pass@1 metric on MiniF2F using the pre-trained ReProver model:

```bash
# Example command - adjust paths based on repo contents
# Note: you might need to run `module load cuda`, or make sure CUDA is available.
python prover/evaluate.py \
	--data-path data/leandojo_benchmark_4/random \
	--gen_ckpt_path kaiyuy/leandojo-lean4-tacgen-byt5-small \
	--num-sampled-tactics 5 \
	--num-theorems 50
# Should take about 60 mins on a machine with GPUs.
# Expected Final Pass@1 value should be around 0.24
```

## 6. Retrieval-Augmented Generation (RAG)
To use retrieval-augmented generation, you first need to index the retrieval corpus:

```bash
python retrieval/index.py \
    --ckpt_path kaiyuy/leandojo-lean4-retriever-byt5-small \
    --corpus-path data/leandojo_benchmark_4/corpus.jsonl \
    --output-path data/leandojo_benchmark_4/indexed_corpus.pkl \
    --batch-size 16
```
> Note: You can reduce the `--batch-size` to reduce GPU memory required if needed.

After indexing is completed, you can run the evaluation with retrieval:

```bash
python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random/ \
    --gen_ckpt_path kaiyuy/leandojo-lean4-tacgen-byt5-small \
    --ret_ckpt_path kaiyuy/leandojo-lean4-retriever-byt5-small \
    --indexed-corpus-path data/leandojo_benchmark_4/indexed_corpus.pkl \
    --num-sampled-tactics 5 \
    --num-theorems 50
```
# Expected Final Pass@1 value should be around 0.18

## 7. Summary of Results

| Configuration | Num Tactics | Num Theorems | Pass@1 |
|---------------|-------------|--------------|--------|
| Non-retrieval | 5           | 50           | 0.24   |
| Retrieval     | 5           | 50           | 0.18   |
| Non-retrieval | 64          | 200          | 0.3990 |
| Retrieval     | 64          | 200          | 0.3586 |

## 8. Logging
ReProver uses `loguru` to capture detailed execution traces, search steps, and debugging information.

### Log Locations
- **Console:** By default, `INFO` level logs are printed to `stderr`.
- **File:** Detailed traces (including `DEBUG` level) are saved to the `logs/` directory by default.
  - When running `prover/evaluate.py`, a new log file is created for each run: `logs/trace_<YYYYMMDD_HHMMSS>.log`.
  - If the `logs/` directory does not exist, it will be created automatically.
  - You can override the log file path by setting the `REPROVER_LOG_FILE` environment variable:
    ```bash
    export REPROVER_LOG_FILE="my_debug_file.log"
    ```

### Log Format
Log entries follow this format:
`YYYY-MM-DD HH:mm:ss | PID:XXXX | LEVEL | Message`

### Tactic Trace Information
For every step in the proof search, the log captures high-resolution data about the interaction between the model and the Lean environment. This is especially useful for understanding why a proof attempt failed.

Each step in the trace includes:
- **THEOREM**: The full name of the theorem being processed.
- **STATE**: The current goal state (tactic state) before a tactic is applied.
- **TACTIC**: The specific tactic suggested by the model.
- **RESULT**: The outcome returned by Lean, including the type of response:
  - `TacticState`: Success, showing the new resulting goal state.
  - `LeanError`: Failure, including the specific error message from Lean (e.g., "unknown identifier", "tactic failed", etc.).
  - `ProofFinished`: The tactic successfully closed the goal.

Example of a trace entry for a failed tactic:
```text
=== STEP ===
[THEOREM]: my_theorem
[STATE]:
n : ℕ
⊢ n + 0 = n
[TACTIC]: induction m
[RESULT (LeanError)]:
unknown identifier 'm'
============
```

### Troubleshooting
* **`lean` not found:** Ensure `source $HOME/.elan/env` is in your `.bashrc` or `.zshrc`.
* **Rate Limit Errors:** Verify `echo $GITHUB_ACCESS_TOKEN` is set correctly.
* **Version Mismatch:** If `lean-dojo` complains about versions, force a cache clear: `rm -rf ~/.cache/lean_dojo`.
