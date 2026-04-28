# AA-LCR × kvpress × gpt-oss-120b — handoff to H200

This experiment was paused on a Blackwell (RTX PRO 6000) box and is being
moved to an H200. Everything you need is in this directory plus a few
sibling paths described below. The runner already supports a Hopper-fast
path (FA3 single-shot prefill); just pass `--attn-impl kernels-community/vllm-flash-attn3`.

## What the experiment is

Replicate Shaoting's prior LMCache-style "quality vs TTFT" trade-off study,
this time on the **AA-LCR (Artificial Analysis Long Context Reasoning)**
benchmark from `huggingface.co/datasets/ArtificialAnalysis/AA-LCR`, using
**openai/gpt-oss-120b** as the target model and **kvpress** for KV-cache
compression.

The flow:

1. **Generate**: for every (context, press, rate) combination, query
   gpt-oss-120b at AA-LCR's ~100k-token contexts. Save a wide CSV row per
   context with columns `base_answer{i}` (uncompressed) and
   `{method}_{rate_suffix}_answer{i}` for each press × rate.
2. **Score** each cell with SQuAD-style token F1 (`evaluate_quality_f1.py`),
   producing one float per cell. Reference is `base_answer{i}` (used by the
   simulator) or `answer{i}` (ground truth, used for the no-garbage check).
3. **Simulate** a multi-tier KV-cache storage policy
   (`StorageManager` from `/home/shaoting/end_to_end/components.py`):
   per-context the policy picks (device, rate, method) that maximizes
   `α · score − TTFT(device, tokens, rate)`, sweeping α to trace a
   quality-vs-TTFT curve. Profiling questions are used to estimate
   per-context per-rate score; testing questions are used to compute the
   actual reported quality at the chosen placements.
4. **Plot** quality vs TTFT — one curve per α-sweep ("ours") plus baseline
   curves (per press at each rate, plus a "prefill always" baseline).

Per-rate scoring inputs come from running `--use-base` against `output_*.csv`,
so each per-rate cell is `F1(compressed_answer, uncompressed_answer)`. The
rate=0 cells become 1.0 by definition; that's the simulator's `score_0p0`.

## Paths inventory (this directory)

```
experiments/aa_lcr/
├── README.md                        — short pointer
├── HANDOFF.md                       — this file
├── AA-LCR_Dataset.csv               — original 100-question CSV from HF
├── profiling.csv                    — 48 rows (questions), 25 contexts shared with testing
├── testing.csv                      — 47 rows (questions), 25 contexts shared with profiling
├── split_by_context.py              — how we built profiling/testing from AA-LCR_Dataset.csv
├── run_aa_lcr_gptoss.py             — the runner (chunked-eager fallback + FA3 single-shot path)
├── evaluate_no_garbage.py           — SQuAD-F1 vs ground truth on uncompressed answers
├── evaluate_quality_f1.py           — SQuAD-F1 scorer (replaces *_answer columns with float scores)
├── simulate.py                      — port of simulation_multi_tier.ipynb to Python
├── components.py                    — StorageManager + kv_cache_calculator (incl. gpt-oss-120b entry)
├── run_all.sh                       — end-to-end driver (gen → score → simulate); honors $ATTN_IMPL
├── speed_profile_gptoss120b.csv     — approximated linear TTFT model (A, B per device)
├── speed_profile_gptoss120b.md      — derivation notes for the above
└── lcr/                             — AA-LCR document corpus (230 .txt files, ~13 MB)
```

This directory is **self-contained** — `simulate.py` imports the bundled
`components.py`, `run_aa_lcr_gptoss.py` reads `lcr/` next to itself,
`run_all.sh` calls the bundled `evaluate_quality_f1.py`. Only the kvpress
package itself comes from the parent repo (install with
`pip install -e ../..`).

The kvpress library on this branch carries two commits that matter for
this experiment:

- `befa6ea` — Add GPT-OSS support to BasePress and KVzipPress
- `d04ff2a` — Patch SnapKVPress for gpt-oss half-split RoPE (mirror of the
  KVzip fix; without it SnapKV silently produces meaningless attention
  scores at long context)

## Stack assumed

- Python 3.12, fresh venv (`uv venv --python 3.12 .venv-gptoss && source .venv-gptoss/bin/activate`)
- `uv pip install -e /path/to/kvpress-gptoss[eval]`
- `uv pip install kernels` (FA3 needs this)
- `transformers >= 5.2`, `torch >= 2.11`, `triton >= 3.6`
- `kernels-community/vllm-flash-attn3` (HF Hub kernel, auto-fetched on first use)

## What was decided / locked in (don't re-derive)

| thing | value | why |
|---|---|---|
| dtype | `torch.bfloat16` | model ships MXFP4 MoE + bf16 attention; load just sets compute dtype, MoE stays MXFP4 if `kernels` is installed |
| rates per press | `{0.3, 0.6, 0.9}` | three is enough to draw a clean curve (user's call) |
| presses | knorm, keydiff, snapkv | user-selected |
| baseline | "no press at all" (`press=None`), saved in `base_answer{i}` columns | identical to KVzipPress at rate 0 but skips KVzip's reconstruction passes |
| max_new_tokens | 2048 | gpt-oss often spends ~600 tokens on the analysis channel before its final answer at 100k context |
| reasoning_effort | `'low'` | shortens the harmony "analysis" preamble |
| brevity prompt | `"Answer the question directly and concisely. Do not show your reasoning."` appended after END QUESTION | further encourages a short final-channel answer |
| answer extraction | post-process: take everything after the last `assistantfinal` substring (after `skip_special_tokens=True` decode) | the harmony channel marker becomes the literal substring after special tokens are stripped |
| quality metric | SQuAD token F1, vs base_answer (for simulator) and vs ground truth (for the "no garbage" sanity check) | user's call: paper-grade and simple |
| singleton contexts | 5 sets with k=1 questions (`co_dc_4Q23`, `co_dc_ann_sup_b`, `ind_clean`, `ind_fow`, `sur_con`) are dropped | user wanted "context appears in both halves" — singletons can't satisfy that |
| storage params (CPU/SSD GB, α-sweep) | TBD — `simulate.py` picks them automatically based on total context size; tweak as needed for a clean knee |

## Non-decisions / things we didn't get to

- F1 scoring on the outputs (script is in `/home/shaoting/openai_generated_questions/evaluate_quality_f1.py` — invoke twice: once with `--use-base`, once without).
- No-garbage-in distribution report — `evaluate_no_garbage.py` is ready, hasn't been run because we don't have a finished output_profiling.csv yet.
- Running the simulation. `simulate.py` is a port of the relevant notebook cells; should run end-to-end once scores_*.csv exist. CPU/SSD/α defaults are heuristic; expect to retune them if the curve is degenerate.

## Why we paused on Blackwell

gpt-oss-120b's attention has features (attention sinks `s_aux`, half-split RoPE)
that transformers ships only against the Hopper-only FA3 kernel. On Blackwell
SM 12.0 with PyTorch 2.11 + Triton 3.6, we exhausted every backend that
the gpt_oss config allows:

- `eager` materializes the full N x N attention scratch at 100k tokens
  (≈1.5 TB). OOMs.
- `flex_attention` with the Triton template miscomputes (we verified end-to-end:
  correct output at 71 tokens, increasingly garbage from ~230 tokens up,
  pure newlines from ~3k tokens up).
- The standard transformers `flash_attention_2` is blacklisted by gpt_oss's
  config (`Only kernels-community/vllm-flash-attn3 is supported`).
- `kernels-community/vllm-flash-attn3` itself errors with `S aux is currently
  only supported for Hopper GPUs`.

The workaround we built was **chunked-eager prefill across all 8 GPUs in
pipeline-parallel**: each chunk's per-layer attention scratch is bounded
to `chunk_size × cumulative_kv_len × 64 heads × 2 bytes`, which fits, and
eager attention is reference-correct. Verified end-to-end with a smoke at
112k tokens producing a coherent on-topic answer. **Wall-clock was the
killer**: prefill 8 min + decode ~1–4 min per question, ~9 min per (context,
press, rate). Full sweep was estimated at ~37 h per CSV (75 h for both),
which is what motivated the H200 move.

On Hopper with FA3 the same runner finishes prefill in seconds; the full
sweep should take **single-digit hours**. Just pass:

```bash
--attn-impl kernels-community/vllm-flash-attn3
```

The runner will detect this and switch to single-shot prefill (skip
chunking). Chunked-eager remains the default fallback.

## Known gotchas

- gpt-oss alternates `sliding_attention` (window=128) and `full_attention`
  layers; kvpress (this branch) compresses only full-attention layers.
  After compressing full-attn layers the cache is shorter than
  sliding-window layers' cache → attention-mask shape mismatch. The runner
  truncates sliding-window cache to match (keeping the most recent tokens —
  the only ones the sliding-window kernel actually attends to anyway).
  This logic is in `_truncate_sliding_window_to_match`. **Should still
  apply on Hopper FA3** (the layer-type alternation is a model property,
  not a backend property).
- `extract_final_channel` regex relies on the literal `assistantfinal`
  substring that appears after special tokens are stripped from the
  harmony decode. Don't change `skip_special_tokens=True` in
  `ChunkedDriver.decode` without also revisiting this.
- The runner's `--resume` reads existing `output_*.csv`'s
  `(document_category, document_set_id)` columns to skip already-completed
  groups. A row is only written when ALL 10 (press, rate) combos for that
  group complete; if you kill mid-group, the partial work is lost.
- `position_ids` for the question forward must continue from the
  **original** context length (`ctx_ids.shape[1]`), not from the
  post-compression `cache.get_seq_length()` — the kept K/V tensors retain
  their original RoPE-baked positions.

## How to set up on H200

```bash
# 1. clone the repo (this branch has GPT-OSS-aware kvpress + the experiment)
git clone -b aa-lcr-experiment https://github.com/Shaoting-Feng/kvpress.git
cd kvpress

# 2. fresh venv + install
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[eval]"
uv pip install kernels             # FA3 hub kernel needs this
uv pip install hf_transfer         # faster model download

# 3. download gpt-oss-120b (≈63 GB MXFP4 + bf16; ~15 min on a fast network)
HF_HUB_ENABLE_HF_TRANSFER=1 hf download openai/gpt-oss-120b

# 4. cd into the experiment
cd experiments/aa_lcr
```

## How to run

End-to-end:

```bash
cd experiments/aa_lcr

# 1. Generate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python run_aa_lcr_gptoss.py \
    --in profiling.csv \
    --out output_profiling.csv \
    --attn-impl kernels-community/vllm-flash-attn3 \
    --resume

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python run_aa_lcr_gptoss.py \
    --in testing.csv \
    --out output_testing.csv \
    --attn-impl kernels-community/vllm-flash-attn3 \
    --resume

# 2. No-garbage check (vs ground truth) on profiling
python evaluate_no_garbage.py --in output_profiling.csv \
    --out scores_profiling_vs_gt.csv

# 3. F1 scoring vs uncompressed (used by simulator)
python evaluate_quality_f1.py --in output_profiling.csv --out scores_profiling.csv --use-base
python evaluate_quality_f1.py --in output_testing.csv  --out scores_testing.csv  --use-base

# 4. Simulate + plot
python simulate.py
```

Or just `ATTN_IMPL=kernels-community/vllm-flash-attn3 ./run_all.sh`.

## Smoke-test recipe (run this first on H200, ~5 min)

```bash
cd experiments/aa_lcr
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python run_aa_lcr_gptoss.py \
    --in profiling.csv \
    --out logs/smoke_output.csv \
    --max-groups 1 \
    --combos "base,knorm:0.3,snapkv:0.3,keydiff:0.3" \
    --attn-impl kernels-community/vllm-flash-attn3
```

Inspect `logs/smoke_output.csv` — `base_answer1`, `knorm_0p3_answer1`,
`snapkv_0p3_answer1`, `keydiff_0p3_answer1` should all be short
on-topic strings (paper names, dates, numbers — depending on the question).
On Blackwell smoke takes 9 min per combo; on Hopper FA3 it should be a
small fraction of that. If smoke is correct, kick off the full
profiling+testing sweeps.

## What success looks like

`intermediate_results/gpt-oss-120b/quality_vs_ttft.png` showing several
curves: an "ours" curve from `simulate.py`'s α-sweep, plus baseline curves
per (press, rate). The "ours" curve should dominate — at any given TTFT
budget it should hit higher quality than any single press at a fixed rate.

If the curves look noisy / degenerate, the usual fix is to retune
`SIZE_CPU_GB`, `SIZE_SSD_GB`, and the `ALPHAS` list in `simulate.py`
(`pick_storage_sizes` and `pick_alphas` set sane defaults but they're
heuristic).
