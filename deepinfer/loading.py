from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

from .model import Qwen35


def load_model(
    path: str | Path,
    *,
    device: str | torch.device = "cuda:0",
    dtype: str = "bfloat16",
    quantization: str = "none",
) -> Qwen35:
    path = Path(path)
    raw_config: dict[str, Any] = json.loads((path / "config.json").read_text())
    config = SimpleNamespace(**raw_config.get("text_config", raw_config))
    torch_dtype = getattr(torch, dtype)
    with torch.device("meta"):
        model = Qwen35(config)
        if quantization == "nf4":
            import bitsandbytes as bnb

            for name, layer in list(model.named_modules()):
                if isinstance(layer, nn.Linear) and name != "lm_head":
                    model.set_submodule(
                        name,
                        bnb.nn.Linear4bit(
                            layer.in_features,
                            layer.out_features,
                            bias=False,
                            compute_dtype=torch_dtype,
                            compress_statistics=True,
                            quant_type="nf4",
                        ),
                    )

    pending = dict(model.named_parameters(remove_duplicate=False))
    if config.tie_word_embeddings:
        del pending["lm_head.weight"]
    index = path / "model.safetensors.index.json"
    shards = (
        sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if index.exists()
        else ["model.safetensors"]
    )
    for shard in shards:
        with safe_open(path / shard, framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                name = key.replace("model.language_model.", "model.", 1)
                if name not in pending:
                    continue
                value = tensors.get_tensor(key)
                parent_name, _, attribute = name.rpartition(".")
                parent = model.get_submodule(parent_name)
                if quantization == "nf4" and isinstance(parent, bnb.nn.Linear4bit):
                    value = bnb.nn.Params4bit(
                        value.to(dtype=torch_dtype),
                        requires_grad=False,
                        compress_statistics=True,
                        quant_type="nf4",
                        module=parent,
                    ).to(device)
                else:
                    value = nn.Parameter(
                        value.to(device=device, dtype=torch_dtype), requires_grad=False
                    )
                setattr(parent, attribute, value)
                del pending[name]
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    return model.eval()
