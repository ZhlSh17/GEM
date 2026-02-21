import os
from typing import Any

from datasets import load_dataset as _load_dataset_hf, DatasetDict

from gem.utils import HF_DATASETS_DIR

TASKS = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "leetcode",
    "pony",
    "aops",
    "theoremqa_questions",
    "theoremqa_theorems",
]


def load_dataset(
    name: str,
    *,
    cache_dir: str | os.PathLike = HF_DATASETS_DIR,
    **kwargs: Any,
) -> DatasetDict:
    return _load_dataset_hf("xlangai/bright", name, cache_dir=cache_dir, **kwargs)
