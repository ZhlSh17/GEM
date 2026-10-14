import json
from time import perf_counter

import torch
from loguru import logger
from tqdm import tqdm
from datasets import concatenate_datasets, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from gem.run.bright import TASKS as BRIGHT_TASKS
from gem.datasets.bright import load_dataset as load_bright_dataset
from gem.model import GEM
from gem.utils import DATA_DIR

# Copied from BRIGHT: https://github.com/xlang-ai/BRIGHT/blob/main/reason.py#L158
BRIGHT_PROMPT = (
    "{query}\n\n"
    "Instructions:\n"
    "1. Identify the essential problem.\n"
    "2. Think step by step to reason and describe what information could be relevant and helpful to address the questions in detail.\n"
    "3. Draft an answer with as many thoughts as you have.\n"
)


class CLMHelper:

    def __init__(self, model_name: str) -> None:
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", dtype=torch.bfloat16
        ).eval()
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)

    def generate(
        self,
        prompts: list[str],
        batch_size: int = 1,
        do_sample: bool = False,
        max_new_tokens: int = 1024,
        **kwargs,
    ) -> list[str]:
        if isinstance(prompts, str):
            prompts = [prompts]

        resps = []
        for start_idx in tqdm(range(0, len(prompts), batch_size), desc="generate"):
            resps.extend(
                self._generate(
                    prompts[start_idx : start_idx + batch_size],
                    do_sample=do_sample,
                    max_new_tokens=max_new_tokens,
                    **kwargs,
                )
            )

        return resps

    def _generate(
        self,
        prompts: list[str],
        do_sample: bool = False,
        max_new_tokens: int = 1024,
        **kwargs,
    ) -> list[str]:
        prompts = self.tokenizer.apply_chat_template(
            [[{"role": "user", "content": x}] for x in prompts],
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        return self.tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def _offline_gen_bright(
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    batch_size: int = 16,
    max_tokens: int = 8192,
) -> dict:
    model = CLMHelper(model_name)

    dataset = load_bright_dataset("examples")

    results = {}

    for task in BRIGHT_TASKS:
        subset = dataset[task]
        queries: list[str] = list(subset["query"])

        resps = model.generate(
            [BRIGHT_PROMPT.format(query=query) for query in queries],
            batch_size=batch_size,
            do_sample=False,
            max_new_tokens=max_tokens,
        )
        results[task] = resps

    return results


def _load_test_samples():
    dataset = load_bright_dataset("examples")
    subsets = []
    for task in BRIGHT_TASKS:
        subset: Dataset = dataset[task]
        subsets.append(subset)

    dataset: Dataset = concatenate_datasets(subsets)
    dataset = dataset.shuffle(seed=42).take(128)
    return dataset


def _benchmark_time(checkpoint: str):
    reasoner = CLMHelper("Qwen/Qwen3-4B-Instruct-2507")
    encoder = GEM.from_pretrained(checkpoint, device_map="auto", dtype=torch.bfloat16).eval()

    dataset = _load_test_samples()

    latency_gen = []
    latency_emb = []

    for idx in tqdm(range(len(dataset))):
        start = perf_counter()
        item = dataset[idx]
        resp = reasoner.generate(
            BRIGHT_PROMPT.format(query=item["query"]),
            batch_size=1,
            do_sample=False,
            max_new_tokens=8192,
        )[0]

        latency_gen.append(perf_counter() - start)

        start = perf_counter()
        embs = encoder.encode_corpus(resp, batch_size=1)
        latency_emb.append(perf_counter() - start)

    with open(DATA_DIR / "qe/pipeline_latency/baseline.json", "wt") as f:
        json.dump({"latency_gen": latency_gen, "latency_emb": latency_emb}, f)


def _benchmark_time_gem(checkpoint: str, length: int | None = None, embed_only: bool = False):
    from .gem_timer import GEMTimer
    from gem.run.offline_gen_bright import _INSTRUCTION_WITH_LENGTH

    model = GEMTimer.from_pretrained(checkpoint, device_map="auto", dtype=torch.bfloat16).eval()
    dataset = _load_test_samples()

    latency_full = []
    for idx in tqdm(range(len(dataset))):
        start = perf_counter()
        item = dataset[idx]
        if not embed_only:
            embs = model.encode_queries(
                item["query"],
                batch_size=1,
                do_sample=False,
                max_new_tokens=8192,
                instruction=_INSTRUCTION_WITH_LENGTH.format(length=length) if length else None,
            )
        else:
            embs = model.encode_corpus(item["query"], batch_size=1)

        latency_full.append(perf_counter() - start)

    with open(DATA_DIR / f"qe/pipeline_latency/gem_{length=}_{embed_only=}.json", "wt") as f:
        json.dump({"latency_full": latency_full, "latency_emb": model.latency}, f)


def main():
    # results = _offline_gen_bright()

    # with open(DATA_DIR / "qe/bright_qwen3.json", "wt") as f:
    #     json.dump(results, f)

    for length in [None, 64, 256, 512, 1024]:
        _benchmark_time_gem("train_runs/gem_qwen3/checkpoint-500", length)

    _benchmark_time_gem("train_runs/gem_qwen3_msmarco_embed_only/checkpoint-500", embed_only=True)


if __name__ == "__main__":
    main()
