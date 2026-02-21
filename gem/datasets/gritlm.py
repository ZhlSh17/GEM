import os
from typing import Any

from datasets import load_dataset as _load_dataset_hf, DatasetDict, Dataset

from gem.utils import HF_DATASETS_DIR
from gem.datasets.format import map_format


def load_raw_dataset(cache_dir: str | os.PathLike = HF_DATASETS_DIR, **kwargs: Any) -> DatasetDict:
    return _load_dataset_hf("GritLM/MEDI2", cache_dir=cache_dir, **kwargs)


def load_dataset(
    cache_dir: str | os.PathLike = HF_DATASETS_DIR,
    num_proc: int | None = None,
    **kwargs: Any,
) -> Dataset:
    return map_format(
        dataset=load_raw_dataset(cache_dir, **kwargs)["train"],
        data_source="medi2",
        num_proc=num_proc,
    )
