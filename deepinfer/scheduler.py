from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from .sampling import SamplingParams


@dataclass(eq=False)
class Sequence:
    request_id: int
    token_ids: list[int]
    prompt_length: int
    params: SamplingParams
    generator: torch.Generator
    num_computed_tokens: int = 0
    num_scheduled_tokens: int = 0
    slot: int | None = None
    finish_reason: str | None = None
    started_at: float = 0.0
    first_token_at: float = 0.0
    finished_at: float = 0.0

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids[self.prompt_length :]


class Scheduler:
    """Admit waiting requests into free slots; rotate active requests within a token budget."""

    def __init__(self, num_slots: int, token_budget: int, chunk_size: int) -> None:
        self.waiting, self.running = deque(), deque()
        self.free_slots = list(reversed(range(num_slots)))
        self.token_budget, self.chunk_size = token_budget, chunk_size

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> list[Sequence]:
        while self.waiting and self.free_slots:
            seq = self.waiting.popleft()
            seq.slot = self.free_slots.pop()
            self.running.append(seq)
        scheduled, budget = [], self.token_budget
        for _ in range(len(self.running)):
            if budget == 0:
                break
            seq = self.running.popleft()
            self.running.append(seq)
            pending = len(seq.token_ids) - seq.num_computed_tokens
            seq.num_scheduled_tokens = min(pending, self.chunk_size, budget)
            budget -= seq.num_scheduled_tokens
            scheduled.append(seq)
        return scheduled

    def finish(self, seq: Sequence) -> None:
        self.running.remove(seq)
        self.free_slots.append(seq.slot)
