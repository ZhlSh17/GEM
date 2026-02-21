import math
import argparse
from collections import defaultdict, Counter
from pathlib import Path

from tqdm import tqdm
from loguru import logger
from datasets import Dataset, load_from_disk, load_dataset as load_dataset_hf

from gem.syn.doc_gen import DocGenerator
from gem.utils import DATA_DIR, StreamProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["msmarco", "msmarco_instruct", "reasonir_hq"])
    parser.add_argument("--input_dir", type=Path, default=DATA_DIR / "syn")
    parser.add_argument("--output_dir", type=Path, default= DATA_DIR / "train_data")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_proc", type=int, default=4)
    return parser.parse_args()


def _log_stats(dataset: Dataset) -> None:
    source2stats = defaultdict(lambda: defaultdict(int))
    for batch in tqdm(dataset.iter(1000), total=math.ceil(len(dataset) / 1000), desc="log stats"):
        for idx in range(len(batch["query"])):
            source: str = batch["_source"][idx]
            source = source.rsplit("_", maxsplit=1)[0]

            has_reason = batch["reason"][idx] is not None
            has_pos = batch["pos"][idx] is not None
            has_neg = batch["neg"][idx] is not None

            source2stats[source]["total"] += 1
            source2stats[source]["has_reason"] += has_reason

            if not has_reason:
                continue

            source2stats[source]["has_pos"] += has_pos
            source2stats[source]["has_neg"] += has_neg
            source2stats[source]["has_pos_neg"] += has_pos and has_neg
            source2stats[source]["no_pos_no_neg"] += not has_pos and not has_neg

            for key in ("is_pos_syn", "is_neg_syn"):
                val = batch.get(key)
                if val:
                    source2stats[source][key] += val[idx]

    total = Counter()
    for source, stats in source2stats.items():
        total += Counter(stats)
        logger.info(f"{source}: {dict(stats)}")

    logger.info(f"total: {dict(total)}")


def _preprocess(raw_dataset: Dataset, num_proc: int = 4) -> Dataset:
    _log_stats(raw_dataset)

    dataset = raw_dataset.filter(lambda x: x["reason"] and x["pos"] and x["neg"], batched=False, num_proc=num_proc)
    logger.info(f"dataset size: {len(raw_dataset)} -> {len(dataset)} after filtering")

    return dataset


def main():
    args = parse_args()
    num_proc = args.num_proc
    batch_size = args.batch_size

    input_dir: Path = args.input_dir / args.dataset
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    save_path_syn: Path = args.output_dir / f"{args.dataset}_syn"
    if save_path_syn.exists():
        raise FileExistsError(save_path_syn)

    dataset = _preprocess(load_from_disk(input_dir), num_proc=num_proc)
    save_path: Path = args.output_dir / args.dataset
    if not save_path.exists():
        dataset.save_to_disk(save_path)

    gen = DocGenerator(
        model=args.model,
        temperature=0,
        max_tokens=2048,
        gpu_memory_utilization=0.9,
        max_num_seqs=batch_size,
        max_model_len=16384,
        max_num_batched_tokens=131072,
        tensor_parallel_size=1,
        num_pos=1,
        num_neg=1,
        swap_space=16,
    )

    cache_file: Path = args.input_dir / f"{args.dataset}_syn.jsonl"
    StreamProcessor(
        func=lambda batch: gen.generate(batch, use_tqdm=False),
        dataset=dataset,
        save_path=cache_file,
        batch_size=batch_size,
    ).run()

    dataset = load_dataset_hf("json", data_files=str(cache_file))["train"]
    _log_stats(dataset)
    dataset = dataset.filter(lambda x: x["reason"] and x["is_pos_syn"] and x["is_neg_syn"], num_proc=num_proc)
    dataset = dataset.select_columns(["instruction", "query", "reason", "pos", "neg", "_source"])

    dataset.save_to_disk(save_path_syn)


if __name__ == "__main__":
    main()
