# Lean Workbook Setup Notes (Environment-Specific Gotchas)

This document covers issues encountered running the Lean Workbook setup and
evaluation (README §"Lean Workbook Dataset Setup") on a shared HPC cluster
(Purdue Scholar, `scholar-zfs.rcac.purdue.edu:/scratch/scholar`) that are
**not** covered by the main README. None of these are inherent to the
dataset or the code path in general — they are specific to running on a
machine with an old system `curl`, a Lean toolchain <= v4.12, and a
contended shared network filesystem. If your environment differs, some or
all of these may not apply, but the diagnostic approach (checking real
child-process CPU/RSS growth, checking `.ast.json` counts) is broadly useful
whenever the Lean Workbook trace/extraction step seems stuck.

## 1. Mathlib cache download fails 100% (`OpenSSL/3.0.8: error:16000069:STORE routines::unregistered scheme`)

**Symptom:** during `scripts/setup_lean_workbook.py`, the `lake exe cache get`
step reports `Downloaded: 0 file(s) [attempted N/5011 = X%], N failed` for
every single file, with a wall of `OpenSSL/3.0.8: ... STORE routines::unregistered
scheme` errors.

**Root cause:** Mathlib's `Cache` tool checks `curl --version`; when it's
below 7.81 (this cluster ships curl 7.76.1), it downloads a statically-linked
`static-curl` binary from GitHub to `~/.cache/mathlib/curl-<version>` and
**unconditionally prefers that cached binary on every subsequent run**
(`getCurl` in `Cache/IO.lean` just checks `CURLBIN.pathExists`, it does not
re-check the system curl's version once the static one has been downloaded
once). That static binary's bundled OpenSSL 3.0.8 can't resolve its
certificate store on this host, so every HTTPS download it makes fails.

**Fix (one-time, persists in the conda env / home cache):**
```bash
conda install -n ReProver -c conda-forge curl -y   # installs curl >= 7.81 ahead of /usr/bin/curl in PATH
rm -f ~/.cache/mathlib/curl-7.88.1 ~/.cache/mathlib/curl.cfg   # drop the broken cached static binary
```
After this, `hash -r` (or a new shell) picks up the new curl, and mathlib's
cache tool's own version check passes, so it uses the system curl instead of
downloading (and reusing) the broken static one.

## 2. `prover/evaluate.py --dataset lean_workbook` always returns `Pass@1: nan` (0 proved, 0 failed, N discarded)

**Symptom:** every theorem in the log ends with `Cannot find the *.ast.json
file for Theorem(...)`, caught as `DojoInitError`, and every theorem is
silently discarded. `Evaluation done! 0 theorems proved, 0 theorems failed,
50 non-theorems discarded` / `Pass@1: nan`.

**Root cause:** LeanDojo's `ExtractData.lean` (in the installed `lean_dojo`
package) decides whether to trace a file via `shouldProcess`, which checks
for an existing `.olean` under `.lake/build/lib/lean/...` (note the `lean/`
path segment). Lean <= v4.12 (this project pins `v4.11.0`) actually places
oleans under `.lake/build/lib/...` — **no** `lean/` segment. `evaluate.py`
already has a compatibility shim, `_patch_leandojo_extract_data_compat()`,
that patches exactly this kind of path mismatch in three other functions
(`findLean`, `getImports`, `HashMap.find?`/`.get?`) — but it was missing the
patch for `shouldProcess`. Since the constructed olean path never exists,
`shouldProcess` returns `false` for *every* file (including Mathlib's own),
so `ExtractData.lean` silently traces zero files and produces zero
`.ast.json`/`.dep_paths` anywhere.

**Fix:** already applied in `prover/evaluate.py`'s
`_patch_leandojo_extract_data_compat()` (patches
`Path.toBuildDir "lib/lean" relativePath "olean"` →
`Path.toBuildDir "lib" relativePath "olean"` in `shouldProcess`, alongside
the pre-existing patches). If this regresses after a `lean_dojo` package
upgrade (the `.replace()` calls are exact-text matches against the
installed package's source, so they silently no-op if the upstream text
changes), re-derive the fix by diffing `shouldProcess` against the other
already-patched functions in the installed
`lean_dojo/data_extraction/ExtractData.lean`.

**Also:** LeanDojo caches traces per-commit at
`~/.cache/lean_dojo/gitpython-project-<commit>/`. If a trace attempt fails
partway through (e.g. hits bug #2 above, or is killed), that incomplete
cache entry is *not* automatically invalidated — a later retry for the same
commit will silently reuse the broken/incomplete trace. Delete it before
retrying:
```bash
rm -rf ~/.cache/lean_dojo/gitpython-project-<commit>/
```
(the commit hash is in `data/lean_workbook_reprover/val.json`'s `"commit"` field).

## 3. Extraction phase looks hung for 60+ minutes with zero completions

**Symptom:** the tqdm progress bar in the log sits at `0/188` for an hour or
more. `ps` shows ~24 `lean --run ExtractData.lean <file>` child processes,
each with RSS plateaued around 850-900MB and CPU% stuck around 13-16%, with
zero new `.ast.json` files appearing anywhere under
`/tmp/reprover-lean-dojo/`.

**Root cause:** LeanDojo's "cheap trace" (the `build_deps=False` path used
by `prover/proof_search.py`, to avoid a ~45-minute full-Mathlib rebuild) runs
`lake env lean --threads {NUM_PROCS} --run ExtractData.lean noDeps`, where
`NUM_PROCS = min(os.cpu_count(), 32)` (`lean_dojo/constants.py`).
`os.cpu_count()` reports this node's full core count (24 here), regardless
of how much of the shared network filesystem's I/O bandwidth is actually
available. This spawns ~24 concurrent Lean processes that each
independently load the *entire* Mathlib environment from the same
network-mounted scratch filesystem — a thundering-herd pattern that makes
the shared filesystem (not CPU) the bottleneck. Note: plain `nproc` reports
`1` on this cluster, which is *misleading* — it respects the `OMP_NUM_THREADS=1`
env var set here, not actual core availability (`cpuset.cpus.effective`
confirms 24 real cores are allocated).

**Fix:**
```bash
export NUM_PROCS=4   # or another small number; 24 is far too many for this filesystem
```
before running `prover/evaluate.py` (or `scripts/setup_lean_workbook.py`,
though that script's own `lake build` calls aren't affected by this specific
var — it only affects the `ExtractData.lean` extraction step). With
`NUM_PROCS=4`, a 50-theorem val-split run went from "zero progress after 68
minutes" to fully completing in about 25 minutes.

**How to tell a real hang from genuine (slow) progress:** check the `lean
--run ExtractData` **child** process, not the `lake env lean ...` **wrapper**
(the wrapper always sits at 0% CPU in the `wait()` syscall — that's normal,
not a hang indicator). If the child's RSS keeps climbing and its cumulative
CPU-seconds keep increasing (even if CPU% looks low, e.g. 13%, because it's
I/O-bound), it's working. A genuine stall shows flat RSS *and* flat
cumulative CPU-seconds across two checks a minute or more apart — only kill
and restart in that case.

## Known-good, reproducible invocation (as of 2026-07-24)

```bash
source env.sh
conda install -n ReProver -c conda-forge curl -y   # one-time
rm -f ~/.cache/mathlib/curl-7.88.1 ~/.cache/mathlib/curl.cfg   # one-time, only if a prior broken static curl was cached

export NUM_PROCS=4

python scripts/setup_lean_workbook.py
python prover/evaluate.py \
    --data-path data/lean_workbook_reprover --dataset lean_workbook --split val \
    --gen_ckpt_path kaiyuy/leandojo-lean4-tacgen-byt5-small \
    --num-sampled-tactics 5 --num-theorems 50
```

Result obtained on the val split (98 generated / 187 of 200 candidate rows
verified across val+test): baseline (no retrieval, no repair) **Pass@1 =
0.22** (11/50 proved). See `scripts/compare_repair_lean_workbook.sh` for a
reproducible baseline-vs-repair comparison.
