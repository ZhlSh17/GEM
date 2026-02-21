import os
from typing import Any
import functools

from datasets import load_dataset as _load_dataset_hf, Dataset, DatasetDict

from gem.utils import HF_DATASETS_DIR


def load_raw_dataset(cache_dir: str | os.PathLike = HF_DATASETS_DIR, **kwargs: Any) -> DatasetDict:
    return _load_dataset_hf("samaya-ai/msmarco-w-instructions", cache_dir=cache_dir, **kwargs)


def _filter_by_instruction(sample: dict, instructed: bool = False) -> bool:
    return sample["has_instruction"] if instructed else not sample["has_instruction"]


def load_dataset(
    instructed: bool = False,
    cache_dir: str | os.PathLike = HF_DATASETS_DIR,
    num_proc: int | None = None,
    **kwargs: Any,
) -> Dataset:
    dataset = load_raw_dataset(cache_dir=cache_dir, **kwargs)["train"]

    name_map = {"positive_passages": "pos"}
    if instructed:
        name_map["new_negatives"] = "neg"
    else:
        name_map["negative_passages"] = "neg"

    dataset = dataset.rename_columns(name_map)

    dataset = dataset.select_columns(["query", "pos", "neg", "has_instruction"])
    dataset = dataset.filter(lambda x: x["query"] and x["pos"] and x["neg"])
    dataset = dataset.filter(functools.partial(_filter_by_instruction, instructed=instructed), num_proc=num_proc)
    dataset = dataset.remove_columns("has_instruction")

    dataset = dataset.map(
        functools.partial(_adapt_format, data_source="promptriever"),
        with_indices=True,
        batched=False,
        num_proc=num_proc,
        desc="reformat promptriever data",
    )

    return dataset


def _adapt_format(
    item: dict,
    idx: int,
    data_source: str,
) -> dict:
    assert isinstance(item["query"], str)

    ret = {"_source": f"{data_source}_{idx}"}

    for key in ("pos", "neg"):
        ret[key] = [doc["text"] for doc in item[key]]

    return ret
