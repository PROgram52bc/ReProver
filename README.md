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
- **MiniF2F**: Mathematical olympiad problems.
- **VeriBench**: Formal verification benchmarks.

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
| BEST      | APRIL-Repair (T5)  | 5           | 50           | TBD    |
| BEST      | gAPRIL-Repair      | 5           | 50           | 0.22   |

## 10. Logging
Detailed traces are saved to `logs/trace_<YYYYMMDD_HHMMSS>.log`. You can view tactic state, suggested tactics, and Lean's error messages for every step.
