from __future__ import annotations

from itertools import count
from time import perf_counter

import torch

from .cache import HybridCache
from .model import Qwen35
from .sampling import SamplingParams, sample
from .scheduler import Scheduler, Sequence


class ModelRunner:
    def __init__(self, model: Qwen35, num_slots: int, max_length: int) -> None:
        self.model = model.eval()
        weight = model.model.embed_tokens.weight
        self.device = weight.device
        self.cache = HybridCache(model.config, num_slots, max_length, weight.device, weight.dtype)

    @torch.inference_mode()
    def run(self, seq: Sequence) -> torch.Tensor | None:
        if seq.num_computed_tokens == 0:
            seq.started_at = perf_counter()
        start = seq.num_computed_tokens
        end = start + seq.num_scheduled_tokens
        inputs = torch.tensor(seq.token_ids[start:end], device=self.device)
        hidden = self.model(inputs, self.cache, seq.slot)
        return self.model.lm_head(hidden[-1]) if end == len(seq.token_ids) else None


class LLMEngine:
    def __init__(
        self,
        model: Qwen35,
        *,
        max_model_len: int = 1024,
        max_num_seqs: int = 1,
        prefill_chunk_size: int = 64,
        max_num_batched_tokens: int = 64,
    ) -> None:
        self.max_model_len = max_model_len
        self.model_runner = ModelRunner(model, max_num_seqs, max_model_len)
        self.scheduler = Scheduler(max_num_seqs, max_num_batched_tokens, prefill_chunk_size)
        eos = model.config.eos_token_id
        self.eos_token_ids = {eos} if isinstance(eos, int) else set(eos or [])
        self.request_ids = count()

    def _new_sequence(self, token_ids: list[int], params: SamplingParams) -> Sequence:
        generator = torch.Generator(device=self.model_runner.device).manual_seed(params.seed)
        return Sequence(next(self.request_ids), list(token_ids), len(token_ids), params, generator)

    def add_request(self, token_ids: list[int], params: SamplingParams) -> Sequence:
        seq = self._new_sequence(token_ids, params)
        self.scheduler.waiting.append(seq)
        return seq

    @torch.inference_mode()
    def step(self) -> list[Sequence]:
        finished = []
        for seq in self.scheduler.schedule():
            logits = self.model_runner.run(seq)
            seq.num_computed_tokens += seq.num_scheduled_tokens
            if logits is None:
                continue
            token = sample(logits, seq.params, seq.generator)
            now = perf_counter()
            if len(seq.token_ids) == seq.prompt_length:
                seq.first_token_at = now
            seq.token_ids.append(token)
            if token in self.eos_token_ids:
                seq.finish_reason = "stop"
            elif len(seq.output_token_ids) == seq.params.max_tokens:
                seq.finish_reason = "length"
            if seq.finish_reason is not None:
                seq.finished_at = now
                self.model_runner.cache.reset(seq.slot)
                self.scheduler.finish(seq)
                finished.append(seq)
        return finished

    def generate(self, prompts: list[list[int]], params: SamplingParams) -> list[Sequence]:
        seqs = [self._new_sequence(ids, params) for ids in prompts]
        self.scheduler.waiting.extend(seqs)
        while self.scheduler.has_unfinished_requests():
            self.step()
        return seqs
