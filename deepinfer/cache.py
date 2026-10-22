from __future__ import annotations

from types import SimpleNamespace

import torch


class HybridCache:
    """Request slots with attention KV, convolution history and FP32 delta state."""

    def __init__(
        self,
        config: SimpleNamespace,
        num_slots: int,
        max_length: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        self.lengths = [0] * num_slots
        self.kv, self.conv, self.recurrent = {}, {}, {}
        for layer, kind in enumerate(config.layer_types):
            if kind == "full_attention":
                shape = (num_slots, config.num_key_value_heads, max_length, config.head_dim)
                self.kv[layer] = tuple(
                    torch.empty(shape, device=device, dtype=dtype) for _ in range(2)
                )
            else:
                channels = (
                    2 * config.linear_num_key_heads * config.linear_key_head_dim
                    + config.linear_num_value_heads * config.linear_value_head_dim
                )
                self.conv[layer] = torch.zeros(
                    num_slots,
                    channels,
                    config.linear_conv_kernel_dim - 1,
                    device=device,
                    dtype=dtype,
                )
                self.recurrent[layer] = torch.zeros(
                    num_slots,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                    device=device,
                    dtype=torch.float32,
                )

    def reset(self, slot: int) -> None:
        self.lengths[slot] = 0
        for layer in self.conv:
            self.conv[layer][slot].zero_()
            self.recurrent[layer][slot].zero_()

    def update_kv(
        self, layer: int, slot: int, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.lengths[slot]
        end = start + key.shape[1]
        keys, values = self.kv[layer]
        keys[slot, :, start:end] = key
        values[slot, :, start:end] = value
        return keys[slot, :, :end], values[slot, :, :end]
