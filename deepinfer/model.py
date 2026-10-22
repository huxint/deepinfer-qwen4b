from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .cache import HybridCache


# Qwen3.5 model layers adapted from Transformers v5.20.2.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, *, gated: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim) if gated else torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        if gate is None:
            return (normalized * (1 + self.weight.float())).to(x.dtype)
        return (self.weight * normalized.to(x.dtype) * F.silu(gate.float())).to(x.dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dim = cos.shape[-1]
    rotated = x[..., :dim]
    half = torch.cat((-rotated[..., dim // 2 :], rotated[..., : dim // 2]), dim=-1)
    return torch.cat((rotated * cos + half * sin, x[..., dim:]), dim=-1)


class Attention(nn.Module):
    def __init__(self, config: SimpleNamespace, layer: int) -> None:
        super().__init__()
        self.layer, self.config = layer, config
        width = config.num_attention_heads * config.head_dim
        kv_width = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, width * 2, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_width, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_width, bias=False)
        self.o_proj = nn.Linear(width, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cache: HybridCache,
        slot: int,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        c, length = self.config, len(x)
        query, gate = self.q_proj(x).view(length, -1, c.head_dim * 2).chunk(2, dim=-1)
        query = self.q_norm(query).transpose(0, 1)
        key = self.k_norm(self.k_proj(x).view(length, -1, c.head_dim)).transpose(0, 1)
        value = self.v_proj(x).view(length, -1, c.head_dim).transpose(0, 1)
        query, key = apply_rope(query, cos, sin), apply_rope(key, cos, sin)
        key, value = cache.update_kv(self.layer, slot, key, value)
        groups = c.num_attention_heads // c.num_key_value_heads
        allowed = torch.arange(key.shape[1], device=x.device)[None, :] <= positions[:, None]
        output = F.scaled_dot_product_attention(
            query,
            key.repeat_interleave(groups, 0),
            value.repeat_interleave(groups, 0),
            attn_mask=allowed,
        )
        output = output.transpose(0, 1).reshape(length, -1) * gate.reshape(length, -1).sigmoid()
        return self.o_proj(output)


class GatedDeltaNet(nn.Module):
    def __init__(self, config: SimpleNamespace, layer: int) -> None:
        super().__init__()
        self.layer, self.config = layer, config
        key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        channels = key_dim * 2 + value_dim
        self.conv1d = nn.Conv1d(
            channels, channels, config.linear_conv_kernel_dim, groups=channels, bias=False
        )
        self.in_proj_qkv = nn.Linear(config.hidden_size, channels, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, config.linear_num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, config.linear_num_value_heads, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(config.linear_num_value_heads))
        self.A_log = nn.Parameter(torch.zeros(config.linear_num_value_heads))
        self.norm = RMSNorm(config.linear_value_head_dim, config.rms_norm_eps, gated=True)
        self.out_proj = nn.Linear(value_dim, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cache: HybridCache, slot: int) -> torch.Tensor:
        c, length = self.config, len(x)
        history = cache.conv[self.layer][slot]
        joined = torch.cat((history, self.in_proj_qkv(x).T), dim=-1)
        history.copy_(joined[:, length:])
        mixed = F.silu(F.conv1d(joined[None], self.conv1d.weight, groups=len(joined)))[0].T
        key_dim = c.linear_num_key_heads * c.linear_key_head_dim
        value_dim = c.linear_num_value_heads * c.linear_value_head_dim
        query, key, value = mixed.split((key_dim, key_dim, value_dim), dim=-1)
        query, key = (
            tensor.reshape(length, c.linear_num_key_heads, c.linear_key_head_dim)
            .repeat_interleave(c.linear_num_value_heads // c.linear_num_key_heads, dim=1)
            .float()
            for tensor in (query, key)
        )
        query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        query = query / c.linear_key_head_dim**0.5
        value = value.reshape(length, c.linear_num_value_heads, c.linear_value_head_dim).float()
        beta = self.in_proj_b(x).sigmoid().float()
        decay = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias)
        state, outputs = cache.recurrent[self.layer][slot], []
        for i in range(length):
            state = state * decay[i].exp()[:, None, None]
            remembered = (state * key[i, :, :, None]).sum(-2)
            delta = (value[i] - remembered) * beta[i, :, None]
            state = state + key[i, :, :, None] * delta[:, None, :]
            outputs.append((state * query[i, :, :, None]).sum(-2))
        cache.recurrent[self.layer][slot].copy_(state)
        output = torch.stack(outputs).to(x.dtype)
        gate = self.in_proj_z(x).reshape_as(output)
        return self.out_proj(self.norm(output, gate).reshape(length, -1))


class MLP(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, config: SimpleNamespace, layer: int) -> None:
        super().__init__()
        self.linear = config.layer_types[layer] == "linear_attention"
        if self.linear:
            self.linear_attn = GatedDeltaNet(config, layer)
        else:
            self.self_attn = Attention(config, layer)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cache: HybridCache,
        slot: int,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.input_layernorm(x)
        output = (
            self.linear_attn(normalized, cache, slot)
            if self.linear
            else self.self_attn(normalized, cache, slot, positions, cos, sin)
        )
        x = x + output
        return x + self.mlp(self.post_attention_layernorm(x))


class Decoder(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(Layer(config, i) for i in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class Qwen35(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        self.model = Decoder(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @torch.inference_mode()
    def forward(self, token_ids: torch.Tensor, cache: HybridCache, slot: int = 0) -> torch.Tensor:
        x = self.model.embed_tokens(token_ids)
        start = cache.lengths[slot]
        positions = torch.arange(start, start + len(token_ids), device=x.device)
        rope = self.config.rope_parameters
        dim = int(self.config.head_dim * rope.get("partial_rotary_factor", 1.0))
        inverse = 1 / (
            rope["rope_theta"] ** (torch.arange(0, dim, 2, device=x.device).float() / dim)
        )
        frequencies = positions.float()[:, None] * inverse
        cos = frequencies.cos().repeat(1, 2).to(x.dtype)
        sin = frequencies.sin().repeat(1, 2).to(x.dtype)
        for layer in self.model.layers:
            x = layer(x, cache, slot, positions, cos, sin)
        cache.lengths[slot] += len(token_ids)
        return self.model.norm(x)
