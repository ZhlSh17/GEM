import argparse
from pathlib import Path
from datasets import load_from_disk, concatenate_datasets, Dataset

from tqdm import tqdm

from gem.utils import DATA_DIR
from gem.datasets.promptriever import load_dataset as load_msmarco


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=DATA_DIR / "train_data", type=Path)
    parser.add_argument("--output_dir", default=DATA_DIR / "train_data/all", type=Path)
    return parser.parse_args()


# for ablation: w/o Doc. Gen.
def _align_docs_for_ablation(raw_dataset: Dataset, syn_dataset: Dataset) -> Dataset:
    key2idx = {}
    for idx, item in enumerate(tqdm(raw_dataset)):
        key = item["query"] + item["reason"]
        key2idx[key] = idx

    def _map_syn(item: dict, idx: int) -> dict:
        key = item["query"] + item["reason"]
        if key not in key2idx:
            raise RuntimeError("expect a superset of `syn_dataset`")

        raw_item = raw_dataset[key2idx[key]]

        item["pos"] = raw_item["pos"]
        item["neg"] = raw_item["neg"]
        return item

    return syn_dataset.map(_map_syn, with_indices=True, batched=False, num_proc=4)


def main():
    args = parse_args()
    input_dir: Path = args.input_dir

    dataset: Dataset = concatenate_datasets(
        [
            # Base 260K (500 steps with 512 batch_size require >= 256,000 samples)
            load_from_disk(input_dir / "msmarco_instruct").shuffle(42).take(160000),
            load_from_disk(input_dir / "msmarco_syn").shuffle(42).take(100000),
            # + Orig. Samples (320K)
            load_msmarco(instructed=False, num_proc=8).shuffle(42).take(30000),
            load_msmarco(instructed=True, num_proc=8).shuffle(42).take(30000),
            # + HQ (370K)
            load_from_disk(input_dir / "reasonir_hq_syn").shuffle(42).take(50000),
        ]
    )

    dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    main()
