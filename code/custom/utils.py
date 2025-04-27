import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation (RMSNorm).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def precompute_freqs_cis(config, seq_len=None):
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Each position in our sequence (1st word, 2nd word, etc.) gets its own set of rotation angles.
    - Position 0 has no rotation (all 1+0j),
    - position 1 has a small rotation,
    - position 2 has more rotation, and so on.
    """
    dim = config.rotary_dim
    seq_len = seq_len if seq_len is not None else config.max_seq_len
    theta = config.rope_theta

    # handle sequence length extrapolation if needed
    if hasattr(config, "original_seq_len") and seq_len > config.original_seq_len:
        base = theta
        factor = config.rope_factor
        beta_fast = config.beta_fast
        beta_slow = config.beta_slow

        # correction functions for sequence extrapolation
        def find_correction_dim(num_rotations, dim, base, max_seq_len):
            return (
                dim
                * math.log(max_seq_len / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
            low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
            high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
            return max(low, 0), min(high, dim - 1)

        def linear_ramp_factor(min, max, dim):
            if min == max:
                max += 0.001
            linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
            ramp_func = torch.clamp(linear_func, 0, 1)
            return ramp_func

        # apply the corrections
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, config.original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    else:
        # without extrapolation
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # compute the complex exponentials
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x, freqs_cis, device=None):
    """
    Applies rotary positional embeddings to the input tensor.
    """
    if device is not None:
        x = x.to(device)
        freqs_cis = freqs_cis.to(device)

    dtype = x.dtype

    seq_len = x.size(2)

    if freqs_cis.size(0) < seq_len:
        raise ValueError(
            f"freqs_cis sequence length {freqs_cis.size(0)} is shorter than input sequence length {seq_len}"
        )

    freqs_cis = freqs_cis[:seq_len]
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # we need to align it with the shape of x_complex, which is (batch, heads, seq_len, dim//2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)

    # make sure dimensions match before multiplication
    if x_complex.size(-1) != freqs_cis.size(-1):
        dim = x_complex.size(-1)
        if freqs_cis.size(-1) > dim:
            freqs_cis = freqs_cis[..., :dim]
        else:
            # pad with 1+0j if freqs_cis is smaller (identity in complex multiplication)
            pad_size = dim - freqs_cis.size(-1)
            identity_pad = torch.polar(
                torch.ones((*freqs_cis.shape[:-1], pad_size), device=device),
                torch.zeros((*freqs_cis.shape[:-1], pad_size), device=device),
            )
            freqs_cis = torch.cat([freqs_cis, identity_pad], dim=-1)

    # apply complex multiplication and view as real
    x = torch.view_as_real(x_complex * freqs_cis).flatten(3).type(dtype)
    return x
