import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np

from custom.utils import *


class BidirectionalSelfAttention(nn.Module):
    """
    Self-attention layer with bidirectional attention and rotary positional embeddings (RoPE)
    """

    def __init__(self, config):
        super().__init__()
        assert config.model.n_embd % config.model.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            config.model.n_embd, 3 * config.model.n_embd, bias=config.model.bias
        )
        # output projection
        self.c_proj = nn.Linear(
            config.model.n_embd, config.model.n_embd, bias=config.model.bias
        )
        # regularisation
        self.attn_dropout = nn.Dropout(config.model.dropout)
        self.resid_dropout = nn.Dropout(config.model.dropout)
        self.n_head = config.model.n_head
        self.n_embd = config.model.n_embd
        self.dropout = config.model.dropout

        # rotary embedding dimension - can not be larger than the head size
        self.head_size = config.model.n_embd // config.model.n_head
        self.rotary_dim = min(config.model.rotary_dim, self.head_size)

        # flash attention only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: Flash Attention not available - using slower attention implementation"
            )
            print(
                "Consider upgrading to PyTorch 2.0 or later to leverage H100 hardware acceleration"
            )

    def forward(self, x, freqs_cis=None, attention_mask=None):
        B, T, C = x.size()  # bs, seq_len, n_embd
        device = x.device

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        # --- RoPE
        if freqs_cis is not None:
            if self.rotary_dim < head_size:
                q_rot, q_pass = (
                    q[..., : self.rotary_dim],
                    q[..., self.rotary_dim :],
                )
                k_rot, k_pass = (
                    k[..., : self.rotary_dim],
                    k[..., self.rotary_dim :],
                )

                q_rot = apply_rotary_emb(q_rot, freqs_cis, device)
                k_rot = apply_rotary_emb(k_rot, freqs_cis, device)

                # concat back
                q = torch.cat([q_rot, q_pass], dim=-1)
                k = torch.cat([k_rot, k_pass], dim=-1)
            else:
                q = apply_rotary_emb(q, freqs_cis, device)
                k = apply_rotary_emb(k, freqs_cis, device)

        if self.flash:
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask.bool()

            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
                scale=1.0 / math.sqrt(k.size(-1)),
            )
        else:
            # attention logits
            attn_logits = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            if attention_mask is not None:
                # for manual attention, 0 means mask and 1 means keep
                attn_logits = attn_logits.masked_fill(attention_mask == 0, -np.inf)

            # attention weights
            attn = F.softmax(attn_logits, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # output projection
        return self.resid_dropout(
            self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))
        )
