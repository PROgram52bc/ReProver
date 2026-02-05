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

### Troubleshooting
* **`lean` not found:** Ensure `source $HOME/.elan/env` is in your `.bashrc` or `.zshrc`.
* **Rate Limit Errors:** Verify `echo $GITHUB_ACCESS_TOKEN` is set correctly.
* **Version Mismatch:** If `lean-dojo` complains about versions, force a cache clear: `rm -rf ~/.cache/lean_dojo`.
