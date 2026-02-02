import os

from datasets import load_dataset as _load_dataset_hf, Dataset

from red.utils import INDEX_DIR


TASKS = [
    "biology",
    "earth_science",
    "economics",
    "pony",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "aops",
    "leetcode",
    "theoremqa_theorems",
    "theoremqa_questions",
]


def load_dataset(
    name: str,
    task: str,
    *,
    cache_dir: str | os.PathLike = INDEX_DIR / "bright",
    **kwargs,
) -> Dataset:
    return _load_dataset_hf("xlangai/bright", name, cache_dir=cache_dir, **kwargs)[task]
