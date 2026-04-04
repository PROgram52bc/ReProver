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
pip install tqdm loguru deepspeed "pytorch-lightning[extra]" transformers wandb openai rank_bm25 git+https://github.com/PROgram52bc/LeanDojo.git vllm datasets
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

### Supported Datasets
- **LeanDojo Benchmark 4**: Standard dataset (default).
- **MiniF2F**: Mathematical olympiad problems. See below for setup instructions.
- **VeriBench**: Formal verification benchmarks.

### MiniF2F Dataset Setup
The MiniF2F dataset in its raw format may not be directly compatible with ReProver's current evaluation scripts. You must transform it into standard JSON arrays before running evaluation.

1. **Download and Transform**:
   ```bash
   python scripts/setup_minif2f_example.py
   ```
   This script downloads `cat-searcher/minif2f-lean4` from Hugging Face and saves `val.json` and `test.json` to the `data/` directory in the required format.

2. **Evaluate**:
   ```bash
   python prover/evaluate.py \
       --data-path data/ \
       --dataset minif2f \
       --gen_ckpt_path kaiyuy/leandojo-lean4-tacgen-byt5-small \
       --num-sampled-tactics 5 \
       --num-theorems 50
   ```

## 5. Running the Baseline (Reproduction)
...
```

## 11. Analysis & Comparison
To compare two different runs (e.g., with and without tactic repair) and see a detailed breakdown of which tactics were proposed and fixed, use the `analyze_repair.py` script:

```bash
python analyze_repair.py baseline.log repair.log --output analysis.csv
```

The script aligns tactics by theorem and goal state, showing:
- Original tactics from the generator.
- Fixed tactics from the repair model.
- Success/failure results for each tactic in both runs.
- Comparison of final theorem outcomes (Solved vs. Failed).

## 12. Summary of Results

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
To use proof repair during search, specify a repair model checkpoint.

### Using the gAPRIL Model (Hugging Face)
```bash
python prover/evaluate.py \
    --data-path data/leandojo_benchmark_4/random \
    --gen_ckpt_path kaiyuy/leandojo-lean4-tacgen-byt5-small \
    --repair-ckpt-path uw-math-ai/gAPRIL-wo-exp \
    --num-sampled-tactics 5 \
    --num-theorems 50
```

## 8. Training with APRIL Dataset
The APRIL dataset contains 260K Lean proof-repair tuples.

### Fast Start (Automated Script)
```bash
chmod +x scripts/train_april.sh
./scripts/train_april.sh
```

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

## 9. Summary of Results
| Algorithm | Configuration | Num Tactics | Num Theorems | Pass@1 |
|-----------|---------------|-------------|--------------|--------|
| BEST      | BEST-Non-retrieval | 5           | 50           | 0.24   |
| BEST      | BEST-Retrieval     | 5           | 50           | 0.34   |
| DFS       | DFS-retrieval      | 5           | 50           | 0.34   |
| BFS       | BFS-retrieval      | 5           | 50           | 0.34   |
| BEST      | BEST-Non-retrieval | 64          | 200          | 0.3990 |
| BEST      | BEST-Non-retrieval | 128         | 200          | 0.4040404040 |
| BEST      | BEST-Retrieval     | 64          | 200          | 0.4394 |
| BEST      | BEST-Non-retrieval | 5           | 200          | 0.2727 |
| BEST      | BEST-Retrieval     | 5           | 200          | 0.3384 |
| BEST      | APRIL-Repair (T5)  | 5           | 50           | TBD    |
| BEST      | gAPRIL-Repair      | 5           | 50           | 0.22   |

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

## 10. Logging
Detailed traces are saved to `logs/trace_<YYYYMMDD_HHMMSS>.log`. You can view tactic state, suggested tactics, and Lean's error messages for every step.
