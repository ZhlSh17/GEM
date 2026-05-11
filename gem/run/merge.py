from datasets import load_from_disk, concatenate_datasets, Dataset

from tqdm import tqdm

from gem.utils import DATA_DIR
from gem.datasets.promptriever import load_dataset as load_msmarco


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


def _merge_other_data(name: str = "") -> None:
    # e.g. name = "llama"
    data_dir = DATA_DIR / f"train_data/{name}"
    dataset: Dataset = concatenate_datasets(
        [
            load_from_disk(data_dir / "msmarco_instruct").shuffle(42).take(160000),
            load_from_disk(data_dir / "msmarco_syn").shuffle(42).take(100000),
            load_msmarco(instructed=False, num_proc=8).shuffle(42).take(30000),
            load_msmarco(instructed=True, num_proc=8).shuffle(42).take(30000),
            load_from_disk(data_dir / "reasonir_hq_syn").shuffle(42).take(50000),
        ]
    )

    dataset.save_to_disk(data_dir / "all")


def _merge_qwen3_data():
    # 500 steps with 512 batch_size require >= 256,000 samples

    # GEM (final)
    dataset: Dataset = concatenate_datasets(
        [
            load_from_disk(DATA_DIR / "train_data/msmarco_instruct").shuffle(42).take(160000),
            load_from_disk(DATA_DIR / "train_data/msmarco_syn").shuffle(42).take(100000),
            load_msmarco(instructed=False, num_proc=8).shuffle(42).take(30000),
            load_msmarco(instructed=True, num_proc=8).shuffle(42).take(30000),
            load_from_disk(DATA_DIR / "train_data/reasonir_hq_syn").shuffle(42).take(50000),
        ]
    )

    dataset.save_to_disk(DATA_DIR / "train_data/all")

    # ablation: w/o HQ and Orig. Samples
    dataset: Dataset = concatenate_datasets(
        [
            load_from_disk(DATA_DIR / "train_data/msmarco_instruct").shuffle(42).take(160000),
            load_from_disk(DATA_DIR / "train_data/msmarco_syn").shuffle(42).take(100000),
        ]
    )

    dataset.save_to_disk(DATA_DIR / "train_data/msmarco_mixed_syn_only")

    # ablation: w/o HQ
    dataset: Dataset = concatenate_datasets(
        [
            load_from_disk(DATA_DIR / "train_data/msmarco_instruct").shuffle(42).take(160000),
            load_from_disk(DATA_DIR / "train_data/msmarco_syn").shuffle(42).take(100000),
            load_msmarco(instructed=False, num_proc=8).shuffle(42).take(30000),
            load_msmarco(instructed=True, num_proc=8).shuffle(42).take(30000),
        ]
    )

    dataset.save_to_disk(DATA_DIR / "train_data/msmarco_mixed")

    # ablation: w/o Orig. Samples
    dataset: Dataset = concatenate_datasets(
        [
            load_from_disk(DATA_DIR / "train_data/msmarco_instruct").shuffle(42).take(160000),
            load_from_disk(DATA_DIR / "train_data/msmarco_syn").shuffle(42).take(100000),
            load_from_disk(DATA_DIR / "train_data/reasonir_hq_syn").shuffle(42).take(50000),
        ]
    )

    # ablation: w/o Doc. Gen.
    dataset: Dataset = concatenate_datasets(
        [
            load_from_disk(DATA_DIR / "train_data/msmarco_instruct").shuffle(42).take(160000),
            _align_docs_for_ablation(
                load_from_disk(DATA_DIR / "train_data/msmarco"),  # superset
                load_from_disk(DATA_DIR / "train_data/msmarco_syn").shuffle(42).take(100000),
            ),
            load_msmarco(instructed=False, num_proc=8).shuffle(42).take(30000),
            load_msmarco(instructed=True, num_proc=8).shuffle(42).take(30000),
            _align_docs_for_ablation(
                load_from_disk(DATA_DIR / "train_data/reasonir_hq"),  # superset
                load_from_disk(DATA_DIR / "train_data/reasonir_hq_syn").shuffle(42).take(50000),
            ),
        ]
    )

    dataset.save_to_disk(DATA_DIR / "train_data/all_no_syn_docs")

    dataset.save_to_disk(DATA_DIR / "train_data/all_no_og_msmarco")

    # embed_only
    dataset: Dataset = concatenate_datasets(
        [
            load_msmarco(instructed=True, num_proc=8).shuffle(42).take(190000),
            load_msmarco(instructed=False, num_proc=8).shuffle(42).take(130000),
        ]
    )

    dataset.save_to_disk(DATA_DIR / "train_data/msmarco_embed_only")

    # embed_only + HQ
    dataset: Dataset = concatenate_datasets(
        [
            load_msmarco(instructed=True, num_proc=8).shuffle(42).take(190000),
            load_msmarco(instructed=False, num_proc=8).shuffle(42).take(130000),
            load_from_disk(DATA_DIR / "train_data/reasonir_hq").shuffle(42).take(50000),
        ]
    )

    dataset.save_to_disk(DATA_DIR / "train_data/all_embed_only")


def main():
    # Final
    dataset: Dataset = concatenate_datasets(
        [
            load_from_disk(DATA_DIR / "train_data/msmarco_instruct").shuffle(42).take(160000),
            load_from_disk(DATA_DIR / "train_data/msmarco_syn").shuffle(42).take(100000),
            load_msmarco(instructed=False, num_proc=8).shuffle(42).take(30000),
            load_msmarco(instructed=True, num_proc=8).shuffle(42).take(30000),
            load_from_disk(DATA_DIR / "train_data/reasonir_hq_syn").shuffle(42).take(50000),
        ]
    )

    dataset.save_to_disk(DATA_DIR / "train_data/all")


if __name__ == "__main__":
    main()
