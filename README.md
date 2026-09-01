# OpenTWBench · B200 (vLLM) runner

Runs the OpenTWBench eval on a modern CUDA box (built for **NVIDIA B200 /
Blackwell**) for the 22 models the Mac/MLX pass left unrun — the big ones (up to
72B), the reasoning distills, and the driver-blocked architectures (mamba
hybrids, LFM2, Qwen3.5) that the old vLLM on the 3090 could not serve. A modern
vLLM on Blackwell serves all of them, fast.

**Protocol is identical to the rest of the suite** — box answer extraction, a
deterministic per-question option shuffle seeded by `sha256(question)`,
temperature 0, the same system prompt, and the box-first + bare-letter lenient
fallback — so the numbers drop straight onto the same leaderboard. **Full
precision only (bfloat16); no quantization anywhere.**

## Quick start

```sh
git clone https://github.com/lianghsun/otb-b200-runner.git
cd otb-b200-runner
HF_TOKEN=hf_xxx bash bootstrap.sh
```

`bootstrap.sh` makes a venv, installs vLLM, runs every model (formosa first,
then the exam), reclaims each model's weights after scoring, and pushes results
to the `b200-results` branch every 5 minutes. It resumes if interrupted — rerun
and it skips finished models and continues partial ones from checkpoint.

Options:

```sh
TP=8 HF_TOKEN=hf_xxx bash bootstrap.sh                       # tensor-parallel over 8 GPUs
MODELS='Qwen/Qwen2.5-72B-Instruct' HF_TOKEN=hf_xxx bash bootstrap.sh   # one model
```

`TP` defaults to the number of visible GPUs. Set it to a divisor of that if you
want to shard a big model across some GPUs and run others in parallel later.

---

## For the operator (or the Claude Code running this) — checklist

1. **HF_TOKEN** — required. It reads the *private* benchmark repo
   (`OpenTWBench/otb-mac-data`) and the gated models. Do **not** print it.

2. **vLLM + Blackwell.** Step 2 of bootstrap installs `vllm` (which pulls a
   matching torch) and then imports it. If the import fails, it's almost always
   a Blackwell/CUDA build mismatch — install a **B200-compatible vLLM+torch**
   (CUDA 12.8+ / sm_100) into `./.venv`, then rerun `bash bootstrap.sh`. Confirm
   `torch.cuda.device_count()` matches the GPUs you expect.

3. **Gated models** — accept the license once on each HF model page **with this
   token's account**, or they FAIL with 403 (logged in `results/_summary.json`,
   not fatal to the batch):
   - `mistralai/Mistral-Nemo-Instruct-2407`
   - `mistralai/Mistral-Small-24B-Instruct-2501`
   - `mistralai/Mixtral-8x7B-Instruct-v0.1`
   - `google/gemma-3-27b-it`
   - `tencent/Hunyuan-7B-Instruct`, `tencent/Hunyuan-A13B-Instruct` (may require it)

4. **Push credentials for `b200-results`.** The repo is cloned over HTTPS
   (public → anonymous), which can pull but not push. To let auto-push work,
   give this box write access to just this repo — a **fine-grained GitHub token**
   scoped to `lianghsun/otb-b200-runner`, `Contents: Read and write`, short
   expiry — then:
   ```sh
   git remote set-url origin https://<TOKEN>@github.com/lianghsun/otb-b200-runner.git
   ```
   Auto-push (and the on-exit push) then land on the `b200-results` branch.
   Rotate/remove the token after the run. If push is not set up, results still
   accumulate locally under `results/` — copy them back by hand.

5. **`trust_remote_code`** is on (Hunyuan and a few others need it). Only models
   from the list below are run.

6. **Reasoning models** (`Phi-4-reasoning-plus`, `QwQ-32B`, the three
   `R1-Distill-*`) get `max_tokens=16384` and a 20k context; they are the slow
   ones. Everything else uses 4096.

## Results

Per-model JSONL lands in `results/<model>/{formosa,exam}.jsonl`, each line
`{qh, ok, boxp, lenp}` — a question hash and three booleans, **no question
text**. A `results/<model>/.done` marker means both benches finished.
`results/_summary.json` lists OK / SKIP / FAILED.

Pushed to the `b200-results` branch, the home box folds them onto the shared
leaderboard (same hash format as the Mac runner — `fold_mac.py`).

## Models (22) — edit `DEFAULT_MODELS` in `eval_vllm.py`

Multimodal: `ornith-ai/Ornith-1.5-9B` (text path).

Big, non-reasoning: `tencent/Hunyuan-7B-Instruct`, `THUDM/glm-4-9b-chat`,
`01-ai/Yi-1.5-9B-Chat`, `mistralai/Mistral-Nemo-Instruct-2407`, `microsoft/Phi-4`,
`Qwen/Qwen3-30B-A3B`, `deepseek-ai/DeepSeek-V2-Lite-Chat`,
`mistralai/Mistral-Small-24B-Instruct-2501`, `google/gemma-3-27b-it`,
`Qwen/Qwen2.5-32B-Instruct`, `zai-org/GLM-4-32B-0414`, `Qwen/Qwen3-32B`,
`01-ai/Yi-1.5-34B-Chat`, `mistralai/Mixtral-8x7B-Instruct-v0.1`,
`Qwen/Qwen2.5-72B-Instruct`, `tencent/Hunyuan-A13B-Instruct`.

Reasoning: `microsoft/Phi-4-reasoning-plus`, `Qwen/QwQ-32B`,
`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`, `-Qwen-32B`, `-Llama-70B`.

If a model's architecture isn't in your vLLM yet it's reported FAILED in the
summary — upgrade vLLM and rerun, or drop it.
