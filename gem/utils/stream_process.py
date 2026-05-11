import os
import json
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING, Union

from tqdm import tqdm

from gem.utils.io import read_last_line

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset


class StreamProcessor:
    """Stream data processing with resumability support.

    `StreamProcessor` iteratively applies `func` to a data batch and incrementally
    saves the outputs into a JSONL file. For resumability, an `index_field`,
    which stores the index of a processed data instance in `dataset`, will be
    added to each of the outputs of `func`.

    Args:
        dataset (Dataset): Hugging Face dataset instance.
        func (Callable[[dict], list[dict]]): Function to be applied for stream processing.
            The input is a batch from `dataset.batch` and the output should be a list of dicts (one for each instance in a batch),
        save_path (str | Path): Path to the existing/resulting JSONL file.
        batch_size (int, optional): Batch size for `dataset.batch`. Defaults to 64.
        index_field (str, optional): Field name for saving the index of a data instance in `dataset`. Defaults to "_idx".
    """

    def __init__(
        self,
        dataset: Union["Dataset", "IterableDataset"],
        func: Callable[[dict], list[dict]],
        save_path: str | Path,
        batch_size: int = 64,
        *,
        index_field: str = "_idx",
    ) -> None:
        self.func = func
        self.dataset = dataset
        self.batch_size = batch_size
        self.save_path = save_path if isinstance(save_path, Path) else Path(save_path)
        self.index_field = index_field

    def run(self) -> None:
        last_idx = -1
        batch_size = self.batch_size

        # resume if possible
        if self.save_path.exists():
            line = read_last_line(self.save_path, encoding="utf-8")

            try:
                last_idx = json.loads(line)[self.index_field]
            except Exception as e:
                with open(self.save_path, "rb") as f:
                    file_size = f.seek(0, os.SEEK_END)

                if file_size != 0:
                    raise RuntimeError(f"error when resume from {self.save_path}") from e
        else:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)

        # batch process
        save_file = open(self.save_path, "at", encoding="utf-8")

        if (last_idx + 1) >= batch_size:
            last_batch_idx = (last_idx + 1) // batch_size - 1
        else:
            last_batch_idx = -1

        for batch_idx, batch in enumerate(tqdm(self.dataset.batch(batch_size=batch_size))):
            if batch_idx <= last_batch_idx:
                continue

            results = self.func(batch)

            for offset, res in enumerate(results):
                idx = batch_idx * batch_size + offset
                if idx <= last_idx:
                    continue

                res[self.index_field] = idx
                save_file.write(json.dumps(res, ensure_ascii=False) + "\n")

            save_file.flush()

        save_file.close()
