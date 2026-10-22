from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 20
    seed: int = 0


def sample(logits: torch.Tensor, params: SamplingParams, generator: torch.Generator) -> int:
    if params.temperature == 0:
        return logits.argmax().item()
    scores = logits.float() / params.temperature
    if params.top_k:
        scores, indices = scores.topk(min(params.top_k, scores.numel()))
    else:
        scores, indices = scores.sort(descending=True)
    probabilities = scores.softmax(dim=-1)
    probabilities.masked_fill_(probabilities.cumsum(-1) - probabilities >= params.top_p, 0)
    return indices[torch.multinomial(probabilities, 1, generator=generator)].item()
