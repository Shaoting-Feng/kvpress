# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import logging
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, Cache, DynamicCache, Pipeline, QuantizedCache
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.base import GenericTensor

from kvpress.presses.base_press import BasePress
from kvpress.presses.decoding_press import DecodingPress
from kvpress.presses.finch_press import FinchPress
from kvpress.presses.key_rerotation_press import KeyRerotationPress
from kvpress.presses.observed_attention_press import ObservedAttentionPress
from kvpress.presses.prefill_decoding_press import PrefillDecodingPress

logger = logging.getLogger(__name__)


class KVPressTextGenerationPipeline(Pipeline):
    """
    KV-Press text generation pipeline without questions:
    - Prefill on the *context* (with KV compression if provided)
    - Collect context prefill logits for tokens 2..N
    - Use last-context logits to seed greedy decoding (no question needed)
    - Return answer + per-step decode logits + context prefill logits
    """

    def _sanitize_parameters(
        self,
        press: Optional[BasePress] = None,
        max_new_tokens: int = 50,
        max_context_length: Optional[int] = None,
        cache: Optional[Cache] = None,
        **kwargs,
    ):
        # Only context is used; no questions.
        if max_context_length is None:
            max_context_length = min(self.tokenizer.model_max_length, int(1e10))

        preprocess_kwargs = {
            "max_context_length": max_context_length,
        }
        forward_kwargs = {
            "press": press,
            "max_new_tokens": max_new_tokens,
            "cache": cache,
        }
        postprocess_kwargs = {}
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(
        self,
        context: str,
        max_context_length: int,
    ):
        """
        Minimal preprocess:
        - Prepend a single BOS if available
        - Encode context only (no chat template, no question)
        """
        # bos_token = getattr(self.tokenizer, "bos_token", "")
        # if bos_token:
        #     context = bos_token + context

        context_ids = self.tokenizer.encode(context, return_tensors="pt", add_special_tokens=False)

        if context_ids.shape[1] > max_context_length:
            logger.warning(
                f"Context length has been truncated from {context_ids.shape[1]} to {max_context_length} tokens."
            )
            context_ids = context_ids[:, :max_context_length]

        # Debug (optional):
        # print("Context token IDs:", context_ids.shape, context_ids[0, :20].tolist())

        return {"context_ids": context_ids}

    def _forward(
        self,
        input_tensors: dict[str, GenericTensor],
        max_new_tokens: int = 50,
        press: Optional[BasePress] = None,
        cache: Optional[Cache] = None,
    ):
        """
        Execute:
        1) Prefill on context (with optional KV compression)
        2) Save context prefill logits (tokens 2..N) and last-context logits
        3) Greedy decode directly from the context (no question)
        """
        context_ids = input_tensors["context_ids"].to(self.model.device)
        context_length = context_ids.shape[1]

        if cache is None:
            cache = DynamicCache()

        # Prefill compression applies (non-decoding presses)
        perform_prefill_compression = press is not None and not isinstance(press, DecodingPress)
        with press(self.model) if perform_prefill_compression else contextlib.nullcontext():
            # Run FULL LM to obtain logits during prefill (so we can export context prefill logits)
            context_position_ids = torch.arange(0, context_length, device=self.model.device).unsqueeze(0)
            outputs_ctx = self.model(
                input_ids=context_ids,
                past_key_values=cache,
                position_ids=context_position_ids,
                output_attentions=self.output_attentions(press),
            )
            # Save context prefill logits (tokens 2..N) to CPU
            self._context_logits = outputs_ctx.logits[0, 1:, :].detach().cpu()
            # Save the last-context-step logits to seed decoding
            self._last_ctx_logits = outputs_ctx.logits[0, -1].detach().cpu()

            logger.debug(f"Context Length: {context_length}")
            logger.debug(f"Compressed Context Length: {cache.get_seq_length()}")

        # Decoding compression applies to decoding or prefill-decoding presses
        perform_decoding_compression = press is not None and isinstance(press, (DecodingPress, PrefillDecodingPress))
        with press(self.model) if perform_decoding_compression else contextlib.nullcontext():
            # If KeyRerotation/Finch with rerotation: context length may change
            if isinstance(press, KeyRerotationPress) or (isinstance(press, FinchPress) and press.rerotate_keys):
                context_length = cache.get_seq_length()

            answer, token_logits = self.generate_from_context(
                cache=cache,
                context_length=context_length,
                max_new_tokens=max_new_tokens,
            )

        return [(answer, token_logits)]

    def _remove_answer_from_cache(self, cache: Cache, cache_seq_lengths: list[int]):
        for layer_idx, sequence_length in enumerate(cache_seq_lengths):
            cache.layers[layer_idx].keys = cache.layers[layer_idx].keys[:, :, :sequence_length]
            cache.layers[layer_idx].values = cache.layers[layer_idx].values[:, :, :sequence_length]

        if isinstance(cache, QuantizedCache):
            for layer_idx, sequence_length in enumerate(cache_seq_lengths):
                cache.layers[layer_idx]._quantized_keys = cache.layers[layer_idx]._quantized_keys[
                    :, :, :sequence_length
                ]
                cache.layers[layer_idx]._quantized_values = cache.layers[layer_idx]._quantized_values[
                    :, :, :sequence_length
                ]

    def generate_from_context(
        self, cache: Cache, context_length: int, max_new_tokens: int
    ) -> Tuple[str, torch.Tensor]:
        """
        Greedy decode starting directly from the context:
        - Seed with last-context logits (collected during prefill)
        - Then generate up to max_new_tokens-1 additional tokens
        Returns:
        (answer, logits_per_step) where logits_per_step has shape [num_steps, vocab_size].
        The first row corresponds to the next-token distribution at the last context position.
        """
        last_ctx_logits = getattr(self, "_last_ctx_logits", None)
        if last_ctx_logits is None:
            raise RuntimeError("No last context logits available. Ensure prefill ran with full LM head.")

        device = self.model.device  # typically 'cuda:0'

        # Collect per-step logits (keep on CPU to save VRAM)
        step_logits: list[torch.Tensor] = [last_ctx_logits]  # first row = next-token at last context pos

        # Seed with the most probable token AFTER the context
        first_id: int = int(last_ctx_logits.argmax().item())
        generated_ids: list[int] = [first_id]

        # position_ids for the next token right after context
        position_ids = torch.tensor([[context_length]], device=device, dtype=torch.long)

        should_stop_token_ids = self.model.generation_config.eos_token_id
        if not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids]

        for i in range(max_new_tokens - 1):
            # Build a CUDA input_ids tensor from the last generated token (python int)
            input_ids = torch.tensor([[generated_ids[-1]]], device=device, dtype=torch.long)

            outputs = self.model(
                input_ids=input_ids,
                past_key_values=cache,
                position_ids=position_ids + i,
            )
            # Take next token and record logits
            next_id = int(outputs.logits[0, -1].argmax().item())
            step_logits.append(outputs.logits[0, -1].detach().cpu())
            generated_ids.append(next_id)

            if next_id in should_stop_token_ids:
                break

        # Decode generated_ids to text
        answer = self.tokenizer.decode(torch.tensor(generated_ids, dtype=torch.long), skip_special_tokens=True)
        logits_tensor = torch.stack(step_logits, dim=0)  # [num_steps, vocab_size]
        return answer, logits_tensor

    def output_attentions(self, press: BasePress):
        if isinstance(press, ObservedAttentionPress):
            return True
        if hasattr(press, "press") and isinstance(press.press, ObservedAttentionPress):
            return True
        return False

    def postprocess(self, model_outputs, *args, **kwargs):
        context_logits = getattr(self, "_context_logits", None)
        ans, logits = model_outputs[0]
        return {"answer": ans, "logits": logits, "context_logits": context_logits}


PIPELINE_REGISTRY.register_pipeline(
    "kv-press-text-generation",
    pipeline_class=KVPressTextGenerationPipeline,
    pt_model=AutoModelForCausalLM,
)
