# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from dataclasses import dataclass

import torch
from torch import nn
import os
import numpy as np

from kvpress.presses.base_press import BasePress

logger = logging.getLogger(__name__)


@dataclass
class ScorerPress(BasePress):
    """
    Base class for score-based KV cache compression methods.

    This class assigns scores to key-value pairs and prune those with the lowest scores.
    Subclasses then implement the `score` method to define how importance is calculated.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    """

    compression_ratio: float = 0.0

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        """
        Compute importance scores for each key-value pair.

        This method must be implemented by subclasses to define how the importance
        of each token position is calculated. Higher scores indicate more important
        tokens that should be kept during compression.

        Parameters
        ----------
        module : nn.Module
            The transformer attention layer where scoring is applied.
        hidden_states : torch.Tensor
            Input embeddings with shape (batch_size, seq_len, hidden_dim).
        keys : torch.Tensor
            Key tensors with shape (batch_size, num_kv_heads, seq_len, head_dim).
        values : torch.Tensor
            Value tensors with shape (batch_size, num_kv_heads, seq_len, head_dim).
        attentions : torch.Tensor
            Attention weights with shape (batch_size, num_heads, seq_len, seq_len).
            May be None if not computed or needed by the scoring method.
        kwargs : dict
            Additional arguments from the forward pass, including cache and position info.

        Returns
        -------
        torch.Tensor
            Importance scores with shape (batch_size, num_kv_heads, seq_len).
            Higher scores indicate more important tokens. The tokens with the
            lowest scores will be pruned during compression.
        """
        raise NotImplementedError

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.compression_ratio == 0:
            return keys, values

        # Compute scores
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        # Get indices of KV pairs with the lowest scores
        k_len = keys.shape[2]
        n_kept = int(k_len * (1 - self.compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices

        from kvpress.presses.observed_attention_press import ObservedAttentionPress
        if isinstance(self, ObservedAttentionPress):
            # IMPRESS
            h0 = torch.unique(indices[0, 0])
            h1 = torch.unique(indices[0, 1])
            h2 = torch.unique(indices[0, 2])
            def jac(a, b):
                inter = torch.sum(torch.isin(a, b))
                union = len(a) + len(b) - inter
                return float(inter) / float(union) if union > 0 else 0.0
            sim = (jac(h0, h1) + jac(h0, h2) + jac(h1, h2)) / 3
            threshold = ((1-self.compression_ratio)/(1+self.compression_ratio))**0.6
            
            # Record positions
            prefix = os.getenv("PREFIX")
            question_index = os.getenv("QUESTION_INDEX")
            filename = prefix + f"q{question_index}/l{kwargs["layer_idx"]}.npy"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            if kwargs["layer_idx"] == 0:
                filename2 = prefix + f"q{question_index}/seq_length.npy"
                os.makedirs(os.path.dirname(filename2), exist_ok=True)
                np.save(filename2, k_len)
            
            if sim >= threshold:
                merged = torch.unique(torch.cat([h0, h1, h2]))
                indices = merged.view(1, 1, -1)
                np.save(filename, merged.detach().cpu().numpy())
            else:
                merged = torch.arange(k_len, device=keys.device)
                np.save(filename, merged.detach().cpu().numpy())
                return keys, values
            indices = indices.repeat(1, keys.shape[1], 1)

        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Prune keys and values
        indices = indices.to(keys.device)
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values
