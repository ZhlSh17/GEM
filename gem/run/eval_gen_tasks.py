import os
import json
import argparse
from pathlib import Path

import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import handle_non_serializable
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

from gem.utils import DATA_DIR
from gem.model import GEM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output_dir", type=Path, default=DATA_DIR / "results/gen_tasks")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--tasks", type=str, nargs="+", default=None)
    parser.add_argument("--gem", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    batch_size = args.batch_size

    tasks = ["ifeval", "mmlu", "arc_challenge", "gsm8k_cot", "humaneval_instruct"] if args.tasks is None else args.tasks
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"  # for coding tasks

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gem:
        model = GEM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto").eval()
        model.forward = model._model.forward
        tokenizer = model.tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto").eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    for task in tasks:
        logger.info(f"{task=}")
        output_file = output_dir / f"{task}.json"
        if output_file.exists():
            logger.warning(f"{output_file} exists; skipped")
            continue

        results = lm_eval.simple_evaluate(
            model=model,
            tasks=[task],
            batch_size=batch_size,
            apply_chat_template=True,
            gen_kwargs={"do_sample": False},
            confirm_run_unsafe_code=True,  # required for coding tasks like humaneval
        )

        with open(output_file, "wt") as f:
            json.dump(results, f, indent=2, default=handle_non_serializable)


if __name__ == "__main__":
    main()
