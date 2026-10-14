import json
import argparse
from pathlib import Path

import torch
from loguru import logger
from datasets import concatenate_datasets, Dataset

from gem.run.bright import TASKS as BRIGHT_TASKS
from gem.datasets.bright import load_dataset as load_bright_dataset
from gem.model import GEM
from gem.exp.gem_timer import GEMTimer

# for experiments with varying generation length
_INSTRUCTION_WITH_LENGTH = """\
Identify the search intent. \
Then think step by step to reason about the criteria that make a document relevant. \
Finally, draft an answer.
Your response should be approximately {length} words.
"""


def _offline_gem_bright(
    checkpoint: str,
    batch_size: int = 16,
    max_tokens: int = 1024,
    instruction: str | None = None,
) -> dict:
    gem = GEM.from_pretrained(checkpoint, device_map="auto", dtype=torch.bfloat16).eval()

    dataset = load_bright_dataset("examples")

    results = {}

    for task in BRIGHT_TASKS:
        subset = dataset[task]
        queries: list[str] = list(subset["query"])
        _, reasons = gem.encode_queries(
            queries,
            instruction=instruction,
            batch_size=batch_size,
            show_progress_bar=True,
            max_length=1024,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_generation=True,
        )
        results[task] = reasons

    results["instruction"] = instruction

    return results


def _benchmark_time_with_resps(
    checkpoint: str | GEMTimer,
    instruction: str | None,
    gem_output_file: str | None,
    batch_size: int = 1,
    num_batches: int = 100,
    max_tokens: int = 8192,
) -> list[float]:
    gem_output = None
    if gem_output_file:
        with open(gem_output_file, "rt") as f:
            gem_output = json.load(f)
            instruction = gem_output["instruction"]

    dataset = load_bright_dataset("examples")

    subsets = []
    for task in BRIGHT_TASKS:
        subset: Dataset = dataset[task]
        if gem_output:
            assert len(gem_output[task]) == len(subset)
            subset = subset.add_column("reason", gem_output[task])

        subsets.append(subset)

    dataset: Dataset = concatenate_datasets(subsets)
    dataset = dataset.shuffle(seed=42)
    queries = dataset["query"]
    reasons = dataset["reason"] if gem_output else None

    num_queries = batch_size * num_batches
    if num_batches > len(queries):
        raise ValueError

    queries = queries[:num_queries]
    if reasons:
        reasons = reasons[:num_queries]

    gem = (
        checkpoint
        if isinstance(checkpoint, GEMTimer)
        else GEMTimer.from_pretrained(checkpoint, device_map="auto", dtype=torch.bfloat16).eval()
    )

    gem.encode_queries(
        queries,
        reasons=reasons,
        instruction=instruction,
        batch_size=batch_size,
        show_progress_bar=True,
        max_length=1024,
        max_new_tokens=max_tokens,
        do_sample=False,
    )

    logger.info(f"avg time elapsed: {sum(gem.latency) / len(gem.latency)}")
    latency = gem.latency
    gem.latency = []
    return latency


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
        reasons: list[str] = list(gpt4_reason[task]["query"])

        if gem_output_file:
            assert len(reasons) == len(gem_reason[task])
            reasons = gem_reason[task]

        for reason in reasons:
            lengths.append(len(tokenizer(reason, add_special_tokens=False)["input_ids"]))

    avg_len = sum(lengths) / len(lengths)
    logger.info(f"\nnum samples: {len(lengths)}\naverage number of tokens: {avg_len}")
    return avg_len


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--output_file", type=Path)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--instruct_with_length", type=int, default=None)
    parser.add_argument("--compute_latency", action="store_true")
    parser.add_argument("--input_file", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    output_file: Path = args.output_file

    if output_file.exists():
        raise FileExistsError(output_file)

    instruction = None  # use default instruction
    if args.instruct_with_length:
        instruction = _INSTRUCTION_WITH_LENGTH.format(length=args.instruct_with_length)

    if args.compute_latency:
        latency = _benchmark_time_with_resps(
            checkpoint=args.checkpoint,
            instruction=instruction,
            gem_output_file=args.input_file,
            batch_size=1,
            num_batches=256,
            max_tokens=8192,
        )
        with open(output_file, "wt") as f:
            json.dump(latency, f)

        return

    results = _offline_gem_bright(
        checkpoint=args.checkpoint,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        instruction=instruction,
    )

    with open(output_file, "wt") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
