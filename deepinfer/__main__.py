from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import torch

from .llm import LLM
from .sampling import SamplingParams


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5 text inference")
    parser.add_argument("--model", required=True, help="Local checkpoint directory")
    parser.add_argument("prompt", help="Prompt text")
    parser.add_argument("--quantization", default="none", help="none or nf4")
    parser.add_argument("--dtype", default="bfloat16", help="bfloat16, float16, or float32")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device, e.g. cuda:0 or cpu")
    parser.add_argument("--max-tokens", type=int, default=64, help="Maximum new tokens")
    parser.add_argument("--max-model-len", type=int, default=1024, help="Context token limit")
    parser.add_argument("--max-num-seqs", type=int, default=1, help="Concurrent request slots")
    parser.add_argument(
        "--prefill-chunk-size", type=int, default=64, help="Prompt tokens per scheduling turn"
    )
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=64, help="Token budget per scheduling turn"
    )
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args(argv)
    llm = LLM(
        args.model,
        quantization=args.quantization,
        dtype=args.dtype,
        device=args.device,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        prefill_chunk_size=args.prefill_chunk_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    output = llm.generate(
        args.prompt, SamplingParams(max_tokens=args.max_tokens), enable_thinking=args.thinking
    )[0]
    print(output.text)
    print(
        f"prompt={output.prompt_tokens}, output={len(output.token_ids)}, "
        f"TTFT={output.time_to_first_token:.2f}s, "
        f"decode={output.decode_tokens_per_second:.2f} tokens/s",
        file=sys.stderr,
    )
    if llm.device.type == "cuda":
        print(
            f"CUDA peak={torch.cuda.max_memory_allocated(llm.device) / 2**30:.2f} GiB",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
