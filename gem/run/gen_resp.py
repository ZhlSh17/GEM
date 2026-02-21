import argparse
from pathlib import Path

from loguru import logger
from datasets import Dataset, load_dataset as load_dataset_hf

from gem.datasets.promptriever import load_dataset as load_msmarco
from gem.datasets.reasonir import load_dataset as load_reasonir
from gem.syn.resp_gen import RespGenerator
from gem.utils import DATA_DIR, StreamProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["msmarco", "msmarco_instruct", "reasonir_hq"])
    parser.add_argument("--output_dir", type=Path, default=DATA_DIR / "syn")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()


def _load_dataset(name: str, num_proc: int = 4) -> Dataset:
    if name == "msmarco":
        return load_msmarco(num_proc=num_proc, instructed=False)
    elif name == "msmarco_instruct":
        return load_msmarco(num_proc=num_proc, instructed=True)
    elif name == "reasonir_hq":
        return load_reasonir("hq", num_proc=num_proc)
    else:
        raise NotImplementedError


def main():
    args = parse_args()
    batch_size = args.batch_size

    gen = RespGenerator(
        model=args.model,
        n=8,
        temperature=1,
        max_tokens=2048,
        gpu_memory_utilization=0.9,
        max_num_seqs=batch_size * 2,
        max_model_len=16384,
        max_num_batched_tokens=131072,
        tensor_parallel_size=1,
        swap_space=16,
    )

    cache_file: Path = args.output_dir / f"{args.dataset}.jsonl"
    StreamProcessor(
        func=lambda batch: gen.generate(batch, use_tqdm=False),
        dataset=_load_dataset(args.dataset),
        save_path=cache_file,
        batch_size=batch_size,
    ).run()

    dataset = load_dataset_hf("json", data_files=str(cache_file))["train"]
    dataset = dataset.select_columns(["instruction", "query", "reason", "pos", "neg", "_source"])

    dataset.save_to_disk(args.output_dir / args.dataset)


if __name__ == "__main__":
    main()
