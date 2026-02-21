import argparse

import mteb
from vllm import LLM

from gem.exp.llm_qe import HyDE, Query2Doc, GEMReasoner
from gem.utils import DATA_DIR
from gem.run.followir import TASK_NAMES as FOLLOWIR_TASKS, preprocess_queries


def _offline_qe_followir(model: LLM) -> None:
    # here we use greedy decoding for fair comparison and reproducibility
    hyde = HyDE(model=model, temperature=0, max_tokens=1024)
    query2doc = Query2Doc(model=model, temperature=0, max_tokens=1024)

    evaluation = mteb.MTEB(tasks=FOLLOWIR_TASKS, task_langs=["en"], do_length_ablation=True)
    evaluation.load_tasks_data()

    all_final_queries = _get_all_followir_queries()
    hyde.generate_and_save(all_final_queries, DATA_DIR / "qe/followir_hyde.json")
    query2doc.generate_and_save(all_final_queries, DATA_DIR / "qe/followir_query2doc.json")


def _offline_gem_followir(checkpoint: str, batch_size: int = 8) -> None:
    gem = GEMReasoner(checkpoint, temperature=0, max_tokens=1024)
    gem.generate_and_save(_get_all_followir_queries(), DATA_DIR / "qe/followir_gem.json", batch_size=batch_size)


def _get_all_followir_queries() -> list[str]:
    evaluation = mteb.MTEB(tasks=FOLLOWIR_TASKS, task_langs=["en"], do_length_ablation=True)
    evaluation.load_tasks_data()

    all_final_queries = []
    for task in evaluation.tasks:
        for attr in ["og_instructions", "changed_instructions", "keywords", "short_instructions"]:
            query2instruction: dict[str, str] = getattr(task, attr)["test"]

            all_final_queries.extend(preprocess_queries(list(query2instruction.keys()), query2instruction))

        # query-only
        all_final_queries.extend(list(task.queries["test"].values()))

    return all_final_queries


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gem", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.gem:
        _offline_gem_followir(args.checkpoint, batch_size=8)
    else:
        model = LLM(
            model="Qwen/Qwen3-4B-Instruct-2507",
            gpu_memory_utilization=0.9,
            max_num_seqs=128,
            max_model_len=16384,
            max_num_batched_tokens=65536,
            swap_space=8,
        )

        _offline_qe_followir(model)

    return


if __name__ == "__main__":
    main()
