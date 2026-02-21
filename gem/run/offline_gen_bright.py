import json
import argparse
from pathlib import Path

import torch
from loguru import logger

from gem.run.bright import TASKS as BRIGHT_TASKS
from gem.datasets.bright import load_dataset as load_bright_dataset
from gem.model import GEM
from gem.exp.gem_timer import GEM_Timer

# for experiments with varying generation length
_INSTRUCTION_WITH_LENGTH = """\
Identify the search intent. \
Then think step by step to reason about the criteria that make a document relevant. \
Finally, draft an answer.
Your response should be approximately {length} words.
"""


def _offline_gem_bright(
    checkpoint: str,
    output_file: str,
    batch_size: int = 16,
    max_tokens: int = 1024,
    instruction: str | None = None,
    compute_latency: bool = False,
) -> None:
    if compute_latency:
        gem = GEM_Timer.from_pretrained(checkpoint, device_map="auto", dtype=torch.bfloat16).eval()
        if batch_size != 1:
            batch_size = 1
            logger.info("using batch_size 1 for computing latency")

    else:
        gem = GEM.from_pretrained(checkpoint, device_map="auto", dtype=torch.bfloat16).eval()

    dataset = load_bright_dataset("examples")

    results = {}
    task2latency = {}

    for task in BRIGHT_TASKS:
        subset = dataset[task]
        queries: list[str] = list(subset["query"])
        _, reasons = gem.encode_query(
            queries,
            instruction=instruction,
            batch_size=batch_size,
            verbose=True,
            max_length=1024,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_generation=True,
        )
        results[task] = reasons

        if compute_latency:
            task2latency[task] = gem.latency
            logger.info(f"({task}) avg time elapsed: {sum(gem.latency) / len(gem.latency)}")
            gem.latency = []

    results["instruction"] = instruction
    if task2latency:
        results["latency"] = task2latency

    with open(output_file, "wt") as f:
        json.dump(results, f, indent=2)


def _compute_avg_reason_len(
    model_name_or_path: str,
    gem_output_file: str | None = None,
) -> float:
    from transformers import AutoTokenizer
    from loguru import logger

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    gpt4_reason = load_bright_dataset("gpt4_reason")
    lengths = []

    # compute for gpt4_reason if `gem_file` is not provided
    if gem_output_file:
        with open(gem_output_file, "rt") as f:
            gem_reason: dict[str, list[str]] = json.load(f)

    for task in BRIGHT_TASKS:
        queries: list[str] = list(gpt4_reason[task]["query"])

        if gem_output_file:
            assert len(queries) == len(gem_reason[task])
            queries = gem_reason[task]

        for query in queries:
            lengths.append(len(tokenizer(query, add_special_tokens=False)["input_ids"]))

    avg_len = sum(lengths) / len(lengths)
    logger.info(f"\nnum queries: {len(lengths)}\naverage number of tokens: {avg_len}")
    return avg_len


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--output_file", type=Path)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--instruct_with_length", type=int, default=None)
    parser.add_argument("--compute_latency", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    output_file: Path = args.output_file

    if output_file.exists():
        raise FileExistsError(output_file)

    instruction = None  # use default instruction
    if args.instruct_with_length:
        instruction = _INSTRUCTION_WITH_LENGTH.format(length=args.instruct_with_length)

    _offline_gem_bright(
        checkpoint=args.checkpoint,
        output_file=output_file,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        instruction=instruction,
        compute_latency=args.compute_latency,
    )


if __name__ == "__main__":
    main()
