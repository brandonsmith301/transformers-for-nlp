import math
import random
import inspect
from dataclasses import dataclass
import os
import json

import torch
import torch.nn as nn
from torch.nn import functional as F

from custom.attention import *
from custom.utils import *
from custom.pool import get_pooling_layer


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(
            config.model.n_embd, 4 * config.model.n_embd, bias=config.model.bias
        )
        self.silu = nn.SiLU()
        self.c_proj = nn.Linear(
            4 * config.model.n_embd, config.model.n_embd, bias=config.model.bias
        )
        self.dropout = nn.Dropout(config.model.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.silu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    """
    Encoder block of the transformer model.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.model.n_embd)
        self.attn = BidirectionalSelfAttention(config)
        self.ln_2 = RMSNorm(config.model.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, freqs_cis=None, attention_mask=None):
        x = x + self.attn(self.ln_1(x), freqs_cis, attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class BaseModel(nn.Module):
    """
    Base Transformer model
    """

    def __init__(self, config):
        super().__init__()
        assert config.model.vocab_size is not None
        assert config.model.max_seq_len is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.model.vocab_size, config.model.n_embd),
                drop=nn.Dropout(config.model.dropout),
                h=nn.ModuleList(
                    [EncoderBlock(config) for _ in range(config.model.n_layer)]
                ),
                ln_f=RMSNorm(config.model.n_embd),
            )
        )

        # pre-compute the rotary embedding frequencies
        # rotary_dim is not > (n_embd // n_head)
        config.model.rotary_dim = min(
            config.model.rotary_dim, config.model.n_embd // config.model.n_head
        )
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(config.model), persistent=False
        )

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        # - nanogpt - https://github.com/karpathy/nanoGPT
        # - BERT did not use this scaling
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.model.n_layer)
                )

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def save_pretrained(self, save_directory, state_dict=None, save_function=None):
        """
        Save model state_dict to a directory.
        """
        os.makedirs(save_directory, exist_ok=True)

        if state_dict is None:
            state_dict = self.state_dict()

        model_path = os.path.join(save_directory, "pytorch_model.bin")
        if save_function is not None:
            save_function(state_dict, model_path)
        else:
            torch.save(state_dict, model_path)

        print(f"Model saved to {save_directory}")

    def _get_encoded_representations(self, input_ids, attention_mask=None):
        """
        Get encoded representations from the transformer model
        """
        device = input_ids.device
        b, t = input_ids.size()
        assert (
            t <= self.config.model.max_seq_len
        ), f"Cannot forward sequence of length {t}, max length is only {self.config.model.max_seq_len}"

        if attention_mask is None:
            attention_mask = torch.ones(b, t, device=device)

        extended_attention_mask = attention_mask.to(dtype=torch.float32)[
            :, None, None, :
        ]

        x = self.transformer.wte(input_ids)  # token embeddings
        x = self.transformer.drop(x)

        freqs_cis = self.freqs_cis.to(device) if hasattr(self, "freqs_cis") else None

        # track all hidden states for pooling layers
        self.all_hidden_states = [x]

        # process through transformer blocks
        for block in self.transformer.h:
            x = block(x, freqs_cis, extended_attention_mask)
            self.all_hidden_states.append(x)

        x = self.transformer.ln_f(x)
        self.all_hidden_states.append(x)

        return x

    def configure_optim(self, device):
        """
        Configure optimisers with weight decay for training.
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.training.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and ("cuda" in str(device))
        extra_args = dict(fused=True) if use_fused else dict()
        print(f"using fused AdamW: {use_fused}")

        return torch.optim.AdamW(
            optim_groups,
            lr=self.config.training.learning_rate,
            betas=self.config.training.betas,
            **extra_args,
        )


class ClassifierModel(BaseModel):
    """
    Encoder-only Transformer model for sequence classification fine-tuning
    """

    def __init__(self, config):
        super().__init__(config)

        self.pooler = get_pooling_layer(config, config)

        self.classifier = nn.Linear(self.pooler.output_dim, config.model.num_classes)

    def _pool_sequence(self, x, attention_mask=None):
        """
        Pool the sequence using the configured pooling strategy
        """
        inputs = {"input_ids": x, "attention_mask": attention_mask}

        return self.pooler(inputs, (x, self.all_hidden_states))

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass for classification.
        """
        x = self._get_encoded_representations(input_ids, attention_mask)
        pooled_output = self._pool_sequence(x, attention_mask)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return logits, loss
