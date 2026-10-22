from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .engine import LLMEngine
from .loading import load_model
from .sampling import SamplingParams


@dataclass
class RequestOutput:
    text: str
    token_ids: list[int]
    prompt_tokens: int
    finish_reason: str
    time_to_first_token: float
    decode_tokens_per_second: float


class LLM:
    def __init__(
        self,
        model: str | Path,
        *,
        quantization: str = "none",
        dtype: str = "bfloat16",
        device: str | torch.device = "cuda:0",
        max_model_len: int = 1024,
        max_num_seqs: int = 1,
        prefill_chunk_size: int = 64,
        max_num_batched_tokens: int = 64,
    ) -> None:
        from transformers import AutoTokenizer

        self.model = load_model(model, quantization=quantization, dtype=dtype, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.engine = LLMEngine(
            self.model,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            prefill_chunk_size=prefill_chunk_size,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        self.device = self.engine.model_runner.device

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        *,
        enable_thinking: bool = False,
    ) -> list[RequestOutput]:
        prompts = [prompts] if isinstance(prompts, str) else prompts
        encoded = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            for prompt in prompts
        ]
        seqs = self.engine.generate(encoded, sampling_params or SamplingParams())
        return [
            RequestOutput(
                text=self.tokenizer.decode(seq.output_token_ids, skip_special_tokens=True),
                token_ids=seq.output_token_ids,
                prompt_tokens=seq.prompt_length,
                finish_reason=seq.finish_reason,
                time_to_first_token=seq.first_token_at - seq.started_at,
                decode_tokens_per_second=(
                    (len(seq.output_token_ids) - 1) / (seq.finished_at - seq.first_token_at)
                    if len(seq.output_token_ids) > 1
                    else 0.0
                ),
            )
            for seq in seqs
        ]
