#!/usr/bin/env python3
"""
AA-LCR runner for gpt-oss-120b with kvpress compression on Blackwell GPUs.

We can't use the standard kvpress text-generation pipeline here because:

  - gpt-oss config blacklists every flash-attn variant except FA3, and FA3
    refuses to run on non-Hopper hardware ("S aux is currently only supported
    for Hopper GPUs").
  - flex_attention's TRITON kernels miscompute on Blackwell SM 12.0 with
    PyTorch 2.11 + Triton 3.6 at long context (verified end-to-end: empty
    output / pure newlines from ~200 tokens upward).
  - eager attention is reference-correct but materializes the full N x N
    attention scratch (~1.5 TB at 100k tokens) on a single layer's GPU, so
    a one-shot prefill OOMs even with device_map='auto'.

So we use eager + device_map='auto' (model weights pipeline-split across all
8 GPUs, ~8 GB each) and prefill the context in chunks. Each chunk's per-layer
attention scratch is bounded by `chunk_size * cumulative_kv_len` rather than
`total_len^2`, which fits in the per-GPU headroom.

For compression we register a *modified* press forward_hook on each
attention layer. The hook respects a `_compress_now` flag so it does nothing
on the intermediate chunks (cache just builds up) and only fires compression
on the final chunk, when the cache holds the full context. That gives the
press the cumulative KV to score and compress against, identical in spirit to
running the kvpress pipeline's one-shot prefill — just paid out across many
chunks to dodge the per-layer N x N OOM.

Sweep:
  - 1 baseline (no compression)        -> base_answer{i}
  - {knorm, keydiff, snapkv} x {0.3, 0.6, 0.9}
                                       -> {method}_{rate_suffix}_answer{i}
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
)
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

from kvpress import (
    KeyDiffPress,
    KnormPress,
    SnapKVPress,
)
from kvpress.utils import extract_keys_and_values

HERE = Path(__file__).resolve().parent
# AA-LCR document corpus, bundled next to this script. Override with
# AA_LCR_ROOT if you keep the corpus elsewhere.
LCR_ROOT = Path(os.environ.get("AA_LCR_ROOT", str(HERE / "lcr")))

PRESS_FACTORIES = {
    "knorm": lambda r: KnormPress(compression_ratio=r),
    "keydiff": lambda r: KeyDiffPress(compression_ratio=r),
    "snapkv": lambda r: SnapKVPress(compression_ratio=r),
}
RATES = [0.3, 0.6, 0.9]
# Generous because at 112k context the model often spends ~600 analysis-channel
# tokens before its final-channel answer, even with reasoning_effort='low'.
MAX_NEW_TOKENS = 2048
DEFAULT_CHUNK = 1024
# When True, do a single-shot prefill (one model() call on the whole
# context) instead of the chunked path. Set automatically when
# --attn-impl=kernels-community/vllm-flash-attn3 is passed: FA3 keeps
# attention scratch O(N), so it doesn't OOM at 100k tokens, and the
# whole prefill runs in seconds.
SINGLE_SHOT_PREFILL_FOR_ATTN = {"kernels-community/vllm-flash-attn3"}
# How often (in chunks) to log progress and force a CUDA cache-empty so the
# allocator doesn't keep growing.
LOG_EVERY_N_CHUNKS = 8

import re as _re_final
# In gpt-oss harmony decode (skip_special_tokens=True), the final-channel
# marker becomes the literal substring "assistantfinal" (the special tokens
# `<|start|>`, `<|channel|>`, `<|message|>` get stripped, leaving the channel
# name "final" run together with the prior turn's "assistant" role token).
# We split on the LAST such marker to recover only the final answer.
# Case-sensitive: the channel name is always lowercase.
_FINAL_RE = _re_final.compile(r"assistantfinal")


def extract_final_channel(text: str) -> str:
    """Return only the final-channel content from a gpt-oss harmony decode."""
    if not text:
        return ""
    matches = list(_FINAL_RE.finditer(text))
    if matches:
        text = text[matches[-1].end():]
    # Strip residual harmony markers / EOS that may leak through.
    for marker in ("<|return|>", "<|end|>", "<|endoftext|>",
                   "<|start|>", "<|channel|>", "<|message|>"):
        text = text.replace(marker, "")
    return text.strip()


def rate_to_suffix(rate: float) -> str:
    return f"{rate:.1f}".replace(".", "p")


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def build_aa_lcr_context(category: str, set_id: str, filenames: list[str]) -> str:
    blocks = []
    base = LCR_ROOT / category / set_id
    for i, fn in enumerate(filenames, start=1):
        path = base / fn
        if not path.exists():
            import unicodedata
            for cand in path.parent.iterdir():
                if unicodedata.normalize("NFC", cand.name) == unicodedata.normalize("NFC", fn):
                    path = cand
                    break
        with path.open(encoding="utf-8") as f:
            doc = f.read()
        blocks.append(f"BEGIN DOCUMENT {i}:\n{doc}\nEND DOCUMENT {i}")
    docs_text = "\n\n".join(blocks)
    return (
        "BEGIN INPUT DOCUMENTS\n\n"
        f"{docs_text}\n\n"
        "END INPUT DOCUMENTS\n\n"
        "Answer the following question using the input documents provided above."
    )


def build_question_prompt(question: str) -> str:
    return (
        "\n\nSTART QUESTION\n\n"
        f"{question}\n\n"
        "END QUESTION\n\n"
        # Direct answer instruction — keeps the final-channel response short
        # and self-contained so token-F1 against AA-LCR ground truth is fair.
        "Answer the question directly and concisely. Do not show your reasoning."
        "\n"
    )


# ---------------------- Chunked prefill driver ---------------------------------


class ChunkedDriver:
    """Holds the model + tokenizer, runs chunked-eager prefill, and applies
    a press's compression on the final chunk via a custom forward hook.

    The standard kvpress press.forward_hook short-circuits when
    `cache_position[-1] > q_len`, which is true on every chunk after the
    first. We replace that hook for the duration of one prefill with a
    flag-gated version: compression runs only when `_compress_now` is True,
    which we set on the final chunk."""

    def __init__(self, model_name: str, dtype: torch.dtype, attn_impl: str = "eager"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.attn_impl = attn_impl
        print(f"[{now_ts()}] [INIT] loading model {model_name} dtype={dtype} "
              f"attn_impl={attn_impl}", file=sys.stderr, flush=True)
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
            device_map="auto",
        )
        self.model.eval()
        self._first_device = next(self.model.parameters()).device
        print(f"[{now_ts()}] [INIT] model ready in {time.time()-t0:.1f}s; "
              f"first param device={self._first_device}", file=sys.stderr, flush=True)
        # Chat template question_suffix probe (rendering once is enough).
        # `reasoning_effort='low'` is the gpt-oss harmony knob that suppresses
        # the long analysis-channel preamble; without it the model spends most
        # of its decode budget on chain-of-thought before producing a final
        # answer (and at MAX_NEW_TOKENS=768 it tends not to reach the final
        # channel at all). This is safe — gpt-oss still answers correctly,
        # just much more concisely.
        sep = "#" * 100
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "X" + sep}],
            add_generation_prompt=True, tokenize=False,
            reasoning_effort="low",
        )
        ctx_part, qsuf = rendered.split(sep)
        self._chat_prefix = ctx_part.replace("X", "")
        self._chat_suffix = qsuf

    def _attention_layers(self):
        language_model = (
            self.model.model.language_model
            if hasattr(self.model.model, "language_model") else self.model.model
        )
        for layer in language_model.layers:
            yield layer

    def _gptoss_skip_layer(self, layer) -> bool:
        # gpt-oss alternates sliding (window=128) and full attention; presses on
        # this branch only apply to full-attention layers (sliding-window layers
        # don't grow with context).
        return (
            isinstance(self.model, GptOssForCausalLM)
            and bool(getattr(layer.self_attn, "sliding_window", None))
        )

    @contextmanager
    def _press_hooks(self, press):
        """Register flag-gated press hooks for all (full-attention) layers.

        The hook calls press.compress() only when self._compress_now is True.
        This lets us run the chunked prefill in two effective phases:
          - chunks 1..N-1: hooks no-op, cache grows
          - chunk N:       hooks compress against the full cumulative cache
        """
        if press is None or getattr(press, "compression_ratio", 0.0) == 0.0:
            yield
            return

        press.post_init_from_model(self.model)
        self._compress_now = False
        rotary_emb = (
            self.model.model.language_model.rotary_emb
            if hasattr(self.model.model, "language_model")
            else self.model.model.rotary_emb
        )
        handles = []

        def hook(module, _input, kwargs, output):
            if not self._compress_now:
                return output
            cache = kwargs["past_key_values"]
            cache_layer = cache.layers[module.layer_idx]
            hidden_states = kwargs["hidden_states"]
            keys, values = extract_keys_and_values(cache, module.layer_idx)
            new_keys, new_values = press.compress(
                module, hidden_states, keys, values, output[1], kwargs
            )
            cache_layer.keys = new_keys
            cache_layer.values = new_values
            return output

        for layer in self._attention_layers():
            if self._gptoss_skip_layer(layer):
                continue
            layer.self_attn.rotary_emb = rotary_emb  # SnapKV needs this
            handles.append(
                layer.self_attn.register_forward_hook(hook, with_kwargs=True)
            )
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def chunked_prefill(self, ctx_ids: torch.Tensor, press, chunk: int) -> DynamicCache:
        import gc
        cache = DynamicCache()
        total = ctx_ids.shape[1]
        # Single-shot prefill on Hopper+FA3: one model() call on the whole
        # context. FA3 is O(N) memory, so the per-layer N x N scratch that
        # forces chunking on Blackwell-eager is not a problem.
        if self.attn_impl in SINGLE_SHOT_PREFILL_FOR_ATTN:
            with self._press_hooks(press):
                self._compress_now = (
                    press is not None
                    and getattr(press, "compression_ratio", 0.0) > 0.0
                )
                t0 = time.time()
                with torch.no_grad():
                    self.model.model(
                        input_ids=ctx_ids.to(self._first_device),
                        past_key_values=cache,
                        use_cache=True,
                    )
                print(f"[{now_ts()}]     [PREFILL] {total} tokens single-shot "
                      f"in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
            self._compress_now = False
            self._truncate_sliding_window_to_match(cache, press)
            return cache

        n_chunks = (total + chunk - 1) // chunk
        with self._press_hooks(press):
            i = 0
            chunk_idx = 0
            t0 = time.time()
            while i < total:
                end = min(i + chunk, total)
                self._compress_now = (
                    end == total
                    and press is not None
                    and getattr(press, "compression_ratio", 0.0) > 0.0
                )
                chunk_ids = ctx_ids[:, i:end].to(self._first_device)
                with torch.no_grad():
                    self.model.model(
                        input_ids=chunk_ids,
                        past_key_values=cache,
                        use_cache=True,
                    )
                # Free transient activations / scratch between chunks. Without
                # this the caching allocator pins blocks that the next chunk's
                # larger attention scratch would otherwise spill into,
                # eventually pinning all 96 GB and slowing every alloc.
                del chunk_ids
                gc.collect()
                torch.cuda.empty_cache()
                i = end
                chunk_idx += 1
                if (chunk_idx % LOG_EVERY_N_CHUNKS == 0) or (i == total):
                    print(f"[{now_ts()}]     [CHUNK] {chunk_idx}/{n_chunks} "
                          f"cache_len={cache.get_seq_length()} "
                          f"elapsed={time.time()-t0:.1f}s",
                          file=sys.stderr, flush=True)
        self._compress_now = False
        self._truncate_sliding_window_to_match(cache, press)
        return cache

    def _truncate_sliding_window_to_match(self, cache: DynamicCache, press) -> None:
        """After per-layer compression on full-attention layers, the
        sliding-window layers (which we don't compress) still hold the full
        prefill length. The model builds the attention mask sized for the
        LARGEST cache, so a short-cache layer hits a shape mismatch in
        `attn_weights + attention_mask`. Truncate every sliding-window
        layer's cache to the same length as the full-attention layers —
        keeping the most recent tokens, which is all the sliding-window
        kernel actually attends to anyway (window_size << full_len for
        gpt-oss-120b)."""
        if press is None or getattr(press, "compression_ratio", 0.0) == 0.0:
            return
        if not isinstance(self.model, GptOssForCausalLM):
            return
        target_len = None
        for layer_idx, layer in enumerate(self._attention_layers()):
            if not getattr(layer.self_attn, "sliding_window", None):
                target_len = cache.layers[layer_idx].keys.shape[2]
                break
        if target_len is None:
            return
        for layer_idx, layer in enumerate(self._attention_layers()):
            if getattr(layer.self_attn, "sliding_window", None):
                cl = cache.layers[layer_idx]
                if cl.keys.shape[2] > target_len:
                    cl.keys = cl.keys[:, :, -target_len:].contiguous()
                    cl.values = cl.values[:, :, -target_len:].contiguous()

    def decode(self, cache: DynamicCache, q_ids: torch.Tensor,
               context_length: int, max_new_tokens: int) -> str:
        device = self._first_device
        q_ids = q_ids.to(device)
        position_ids = torch.arange(
            context_length, context_length + q_ids.shape[1], device=device
        ).unsqueeze(0)
        with torch.no_grad():
            out = self.model(
                input_ids=q_ids,
                past_key_values=cache,
                position_ids=position_ids,
                num_logits_to_keep=1,
            )
        gen_ids = [out.logits[0, -1].argmax()]
        position_ids = position_ids[:, -1:] + 1
        eos = self.model.generation_config.eos_token_id
        if not isinstance(eos, list):
            eos = [eos]
        for i in range(max_new_tokens - 1):
            with torch.no_grad():
                out = self.model(
                    input_ids=gen_ids[-1].unsqueeze(0).unsqueeze(0),
                    past_key_values=cache,
                    position_ids=position_ids + i,
                )
            new_id = out.logits[0, -1].argmax()
            gen_ids.append(new_id)
            if new_id.item() in eos:
                break
        return self.tokenizer.decode(torch.stack(gen_ids), skip_special_tokens=True)

    def trim_cache_back_to(self, cache: DynamicCache, length: int) -> None:
        """Truncate every layer's K/V back to `length` along the seq dim.
        We use this between question decodes so each question starts from
        the same prefilled context cache."""
        for layer_idx in range(len(cache.layers)):
            cl = cache.layers[layer_idx]
            cl.keys = cl.keys[:, :, :length]
            cl.values = cl.values[:, :, :length]

    def encode_context(self, context_text: str) -> torch.Tensor:
        full = self._chat_prefix + context_text + self._chat_suffix.replace(
            self._chat_suffix, ""
        )
        # Replace question_suffix back in: the runner appends it to the
        # question side, not context side, so encode just prefix+context here.
        full = self._chat_prefix + context_text
        return self.tokenizer.encode(full, add_special_tokens=False, return_tensors="pt")

    def encode_question(self, question_prompt: str) -> torch.Tensor:
        return self.tokenizer.encode(
            question_prompt + self._chat_suffix,
            add_special_tokens=False, return_tensors="pt",
        )


# ---------------------- Main driver --------------------------------------------


def build_output_columns(group_sizes: dict[tuple[str, str], int]) -> list[str]:
    n_max = max(group_sizes.values()) if group_sizes else 1
    cols = ["document_category", "document_set_id", "data_source_filenames", "length"]
    for i in range(1, n_max + 1):
        cols.append(f"question{i}")
        cols.append(f"answer{i}")
        cols.append(f"base_answer{i}")
        for method in PRESS_FACTORIES:
            for r in RATES:
                cols.append(f"{method}_{rate_to_suffix(r)}_answer{i}")
    return cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True)
    ap.add_argument("--out", dest="out_csv", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--torch-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn-impl", default="eager",
                    help="Attention implementation. 'eager' (default) is the "
                    "Blackwell-safe fallback paired with chunked prefill. On "
                    "Hopper, pass 'kernels-community/vllm-flash-attn3' for "
                    "FA3 + single-shot prefill (much faster).")
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    ap.add_argument("--max-groups", type=int, default=0,
                    help="If >0, process at most this many groups (smoke test).")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--combos", default="",
                    help="Comma-separated list of combos to run, e.g. "
                    "'base,knorm:0.3,snapkv:0.6'. Empty = run all (baseline + sweep).")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.torch_dtype]

    df_in = pd.read_csv(args.in_csv)
    df_in = df_in.sort_values(
        ["document_category", "document_set_id", "question_id"]
    ).reset_index(drop=True)
    groups = list(df_in.groupby(["document_category", "document_set_id"], sort=False))
    group_sizes = {key: len(g) for key, g in groups}
    if args.max_groups > 0:
        groups = groups[: args.max_groups]
        group_sizes = {key: group_sizes[key] for key, _ in groups}
    cols = build_output_columns(group_sizes)
    print(f"[{now_ts()}] [INIT] groups={len(groups)} "
          f"max_q_per_group={max(group_sizes.values()) if group_sizes else 0}",
          file=sys.stderr, flush=True)

    done_keys = set()
    if args.resume and os.path.exists(args.out_csv) and os.path.getsize(args.out_csv) > 0:
        try:
            tmp = pd.read_csv(args.out_csv, usecols=["document_category", "document_set_id"])
            done_keys = set(zip(tmp["document_category"].astype(str),
                                tmp["document_set_id"].astype(str)))
            print(f"[{now_ts()}] [RESUME] {len(done_keys)} groups already done",
                  file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[{now_ts()}] [RESUME] could not read existing out: {e}",
                  file=sys.stderr, flush=True)

    if not (args.resume and os.path.exists(args.out_csv) and os.path.getsize(args.out_csv) > 0):
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()

    driver = ChunkedDriver(args.model, dtype, attn_impl=args.attn_impl)

    combos: list[tuple[str, float | None]] = []
    if args.combos.strip():
        for tok in args.combos.split(","):
            tok = tok.strip()
            if tok == "base":
                combos.append(("base", None))
            elif ":" in tok:
                m, r = tok.split(":")
                combos.append((m.strip(), float(r)))
    else:
        combos.append(("base", None))
        if not args.baseline_only:
            for method in PRESS_FACTORIES:
                for r in RATES:
                    combos.append((method, r))

    for gi, ((cat, set_id), group) in enumerate(groups, start=1):
        if (str(cat), str(set_id)) in done_keys:
            print(f"[{now_ts()}] [SKIP] {gi}/{len(groups)} {cat}/{set_id}",
                  file=sys.stderr, flush=True)
            continue

        filenames = [s.strip() for s in str(group.iloc[0]["data_source_filenames"]).split(";")]
        context = build_aa_lcr_context(cat, set_id, filenames)
        ctx_ids = driver.encode_context(context)
        ctx_len = ctx_ids.shape[1]

        questions_text = list(group["question"].astype(str))
        answers_text = list(group["answer"].astype(str))
        n_q = len(questions_text)
        question_prompts = [build_question_prompt(q) for q in questions_text]
        question_ids = [driver.encode_question(qp) for qp in question_prompts]

        print(f"[{now_ts()}] [GROUP] {gi}/{len(groups)} {cat}/{set_id} "
              f"q={n_q} ctx_tokens={ctx_len}", file=sys.stderr, flush=True)

        row = {c: "" for c in cols}
        row["document_category"] = cat
        row["document_set_id"] = set_id
        row["data_source_filenames"] = ";".join(filenames)
        row["length"] = ctx_len
        for i, (q, a) in enumerate(zip(questions_text, answers_text), start=1):
            row[f"question{i}"] = q
            row[f"answer{i}"] = a

        for combo in combos:
            method, rate = combo
            if method == "base":
                press = None
                col_prefix = "base"
                tag = "base"
            else:
                press = PRESS_FACTORIES[method](rate)
                col_prefix = f"{method}_{rate_to_suffix(rate)}"
                tag = f"{method}@{rate}"

            t1 = time.time()
            try:
                cache = driver.chunked_prefill(ctx_ids, press, args.chunk)
                # cache_seq_after_prefill is what cache.get_seq_length() will
                # return — used by trim_cache_back_to between questions.
                cache_seq_after_prefill = cache.get_seq_length()
                # context_length passed to decode is the ORIGINAL prefill
                # length: the kept K/V tensors still carry their original
                # RoPE-baked positions, and the new question tokens must
                # continue from that original position, not from the
                # compressed cache index.
                original_ctx_len = ctx_len
                for qi, q_ids in enumerate(question_ids, start=1):
                    raw = driver.decode(cache, q_ids, original_ctx_len, MAX_NEW_TOKENS)
                    final = extract_final_channel(raw)
                    row[f"{col_prefix}_answer{qi}"] = final
                    # restore cache for next question
                    driver.trim_cache_back_to(cache, cache_seq_after_prefill)
                del cache
                torch.cuda.empty_cache()
                print(f"[{now_ts()}]   [DONE] {tag} in {time.time()-t1:.1f}s",
                      file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[{now_ts()}]   [ERR ] {tag}: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                import traceback; traceback.print_exc(file=sys.stderr)
                for qi in range(1, n_q + 1):
                    row[f"{col_prefix}_answer{qi}"] = ""

        with open(args.out_csv, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writerow(row)
        print(f"[{now_ts()}] [WRITE] {cat}/{set_id} appended",
              file=sys.stderr, flush=True)

    print(f"[{now_ts()}] [ALL DONE] groups={len(groups)}",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
