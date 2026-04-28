# speed_profile_gptoss120b.csv — derivation

The CSV is read by `end_to_end/components.py::StorageManager._load_ttft_params`
and supplies `(A, B)` for the linear model

```
TTFT(device, num_context_tokens, rate) = A * num_context_tokens * (1 - rate) + B
```

with `device ∈ {None=prefill, cpu, ssd, remote}`. We do not have a measured
profile for gpt-oss-120b, so this file is **approximated** by scaling the
existing `speed_profile_qwen30b.csv` (Qwen3-30B-A3B-Instruct-2507):

| coefficient | meaning | scaling rule |
|---|---|---|
| `A` for `device=None` (prefill) | FLOPs/token during context prefill | unchanged — gpt-oss-120b has ~5.1B active params (MoE), Qwen3-30B-A3B has ~3B active (MoE); both are MoE with similar compute-per-token magnitude. Treating as the same order is a reasonable first cut. |
| `A` for `cpu`/`ssd`/`remote` | (KV bytes / token) / (tier bandwidth) | scaled by `kv_per_tok(gptoss) / kv_per_tok(qwen30b)` |
| `B` (all rows) | fixed setup overhead | unchanged — assumed bandwidth-independent |

KV bytes per token (full-attention layers only, bf16):

```
gpt-oss-120b: 18 layers * 2 (K,V) * 8 kv_heads * 64 head_dim * 2 B = 36 864 B
qwen3-30b-a3b: 0.0915 / 1000 GB/token = 98 250 B (per components.py constant)
ratio = 36 864 / 98 250 = 0.375
```

(Note: gpt-oss-120b alternates `sliding_attention` / `full_attention` per
layer; `sliding_attention` layers have `sliding_window=128`, so their KV is
constant in context length. Only the 18 `full_attention` layers grow with
context — those are the ones the kvpress presses target on this branch.)

Resulting CSV (Qwen30B → gpt-oss-120b):

| device | A_qwen30b | A_gptoss120b | B (unchanged) |
|---|---|---|---|
| None   | 3.9516E-05 | 3.9516E-05         | -3.0118E-02 |
| cpu    | 3.3988E-06 | 3.3988E-06 × 0.375 = 1.2746E-06 | 5.4185E-02 |
| ssd    | 1.7693E-05 | 1.7693E-05 × 0.375 = 6.6349E-06 | 5.1854E-02 |
| remote | 1.8780E-04 | 1.8780E-04 × 0.375 = 7.0425E-05 | 3.2740E-02 |

This is a placeholder. Replace with measured values when a vLLM+LMCache stack
running gpt-oss-120b is available.
