import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

from . import rotary
from .fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
    get_bias_dropout_add_scale,
    modulate_fused,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(x_skip.view(-1, dim_out), x.view(-1, dim_in), W.T, alpha=residual_scale).view(
        *x.shape[:-1], dim_out
    )


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes
        # New improvement: can think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)  # TODO:?
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout

    def _get_bias_dropout_scale(self):
        return bias_dropout_add_scale_fused_train if self.training else bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, seqlens=None):
        """

        :param x: b, s, D
        :param rotary_cos_sin:
        :param attention_masks: b, s
        :param seqlens:
        :return:
        """
        # Remember here, Huanran remove the dropout. We may readd dropout here.
        batch_size, seq_len = x.shape[0], x.shape[1]
        # step1: attention operation
        x_skip = x
        x = self.norm1(x)
        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads)
        with torch.amp.autocast("cuda", enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        qkv = rearrange(qkv, "b s ... -> (b s) ...")

        if seqlens is None:
            cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device=qkv.device)
        else:
            cu_seqlens = seqlens.cumsum(-1)
        x = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, seq_len, 0.0, causal=True)

        x = rearrange(x, "(b s) h d -> b s (h d)", b=batch_size)

        x = self.attn_out(x) + x_skip

        # step2: mlp operation
        x = self.mlp(self.norm2(x)) + x
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors,
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class SEDDBackboneForGPT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 50257,
        n_heads: int = 12,
        dropout: float = 0.1,  # Huanran: not implemented yet
        n_blocks: int = 12,
    ):
        super().__init__()
        self.vocab_embed = EmbeddingLayer(hidden_size, vocab_size)
        self.rotary_emb = rotary.Rotary(hidden_size // n_heads)
        self.blocks = nn.ModuleList([DDiTBlock(hidden_size, n_heads, dropout=dropout) for _ in range(n_blocks)])
        self.output_layer = DDitFinalLayer(hidden_size, vocab_size)

    def _get_bias_dropout_scale(self):
        return bias_dropout_add_scale_fused_train if self.training else bias_dropout_add_scale_fused_inference

    def forward(self, indices):
        x = self.vocab_embed(indices)

        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, seqlens=None)

            x = self.output_layer(x)

        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))

        return x
