import os
import json
from dataclasses import dataclass
from collections.abc import Callable

from tqdm import tqdm
from loguru import logger

from gem.utils import INDEX_DIR, DATA_DIR, ROOT_DIR
from gem.datasets.bright import TASKS
from gem.datasets.bright import load_dataset


def log_results(
    results_dir: str | os.PathLike,
    model: str,
    long: bool = False,
) -> str:
    task2score = {}
    for task in TASKS:
        path = os.path.join(results_dir, f"{task}_{model}_long_{long}", "results.json")
        if not os.path.exists(path):
            continue

        with open(path, "rt") as f:
            results = json.load(f)
            task2score[task] = results["NDCG@10"] * 100

    if task2score:
        task2score["avg"] = sum(task2score.values()) / len(task2score)
        logger.info(f"\n{json.dumps(task2score, indent=2)}")


_RETRIEVERS = [
    "bm25",
    "cohere",
    "e5",
    "google",
    "grit",
    "inst-l",
    "inst-xl",
    "openai",
    "qwen",
    "qwen2",
    "sbert",
    "sf",
    "voyage",
    "bge",
    "bge_ce",
    "nomic",
    "m2",
    "contriever",
    "reasonir",
    "promptriever",
    "gem",
]


@dataclass
class Args:
    task: str
    model: str
    long_context: bool = False
    query_max_length: int = -1
    doc_max_length: int = -1
    encode_batch_size: int = -1
    output_dir: str = str(DATA_DIR / "results/bright")
    cache_dir: str = str(INDEX_DIR / "bright")
    config_dir: str = str(ROOT_DIR / "third_party/ReasonIR/evaluation/bright/configs")
    checkpoint: str | None = None
    key: str | None = None
    input_file: str | None = None
    reasoning: str | None = None
    debug: bool = False
    ignore_cache: bool = False
    # additional args for GEM:
    overwrite: bool = False
    generation_batch_size: int = -1
    no_generation: bool = False
    gem_reason_file: str | None = None  # pre-computed responses


def _update_retrieval_funcs(funcs: dict[str, Callable]) -> None:
    from gem.datasets.bright.retrievers import retrieval_gem, retrieval_promptriever

    funcs["gem"] = retrieval_gem
    funcs["promptriever"] = retrieval_promptriever


def run_bright(args: Args) -> None:
    """Main.

    Adapted from:
        https://github.com/xlang-ai/BRIGHT/blob/main/run.py
        https://github.com/facebookresearch/ReasonIR/blob/main/evaluation/bright/run.py
    """
    from third_party.ReasonIR.evaluation.bright.retrievers import RETRIEVAL_FUNCS, calculate_retrieval_metrics

    _update_retrieval_funcs(RETRIEVAL_FUNCS)
    reasons: list[str] | None = None
    gem_instruction = None

    args.output_dir = os.path.join(args.output_dir, f"{args.task}_{args.model}_long_{args.long_context}")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    score_file_path = os.path.join(args.output_dir, f"score.json")

    print("loading examples...")
    if args.input_file is not None:
        with open(args.input_file) as f:
            examples = json.load(f)
    elif args.reasoning is not None:
        examples = load_dataset("examples")[args.task]
        if args.reasoning != "gem":
            reasoning_examples = load_dataset(f"{args.reasoning}_reason")[args.task]
            if args.model == "gem":
                reasons = list(reasoning_examples["query"])
            else:
                examples = reasoning_examples
        else:
            if args.gem_reason_file is None:
                raise ValueError("require gem_reason_file")

            with open(args.gem_reason_file, "rt") as f:
                _data: dict = json.load(f)

            reasons = _data[args.task]
            gem_instruction = _data.get("instruction")
    else:
        examples = load_dataset("examples")[args.task]

    print("loading documents...")
    if args.long_context:
        doc_pairs = load_dataset("long_documents")[args.task]
    else:
        doc_pairs = load_dataset("documents")[args.task]

    doc_ids = []
    documents = []
    for dp in doc_pairs:
        doc_ids.append(dp["id"])
        documents.append(dp["content"])

    if not os.path.isfile(score_file_path) or args.overwrite:
        config = None
        default_config_path = os.path.join(args.config_dir, args.model, f"{args.task}.json")
        if os.path.exists(default_config_path):
            with open(default_config_path) as f:
                config = json.load(f)
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

        queries = []
        query_ids = []
        excluded_ids = {}
        for e in examples:
            queries.append(e["query"])
            query_ids.append(e["id"])
            excluded_ids[e["id"]] = e["excluded_ids"]
            overlap = set(e["excluded_ids"]).intersection(set(e["gold_ids"]))
            assert len(overlap) == 0
        assert len(queries) == len(query_ids), f"{len(queries)}, {len(query_ids)}"
        if not os.path.isdir(os.path.join(args.cache_dir, "doc_ids")):
            os.makedirs(os.path.join(args.cache_dir, "doc_ids"))
        if os.path.isfile(os.path.join(args.cache_dir, "doc_ids", f"{args.task}_{args.long_context}.json")):
            with open(os.path.join(args.cache_dir, "doc_ids", f"{args.task}_{args.long_context}.json")) as f:
                cached_doc_ids = json.load(f)
            for id1, id2 in zip(cached_doc_ids, doc_ids):
                assert id1 == id2
        else:
            with open(os.path.join(args.cache_dir, "doc_ids", f"{args.task}_{args.long_context}.json"), "w") as f:
                json.dump(doc_ids, f, indent=2)
        assert len(doc_ids) == len(documents), f"{len(doc_ids)}, {len(documents)}"

        print(f"{len(queries)} queries")
        print(f"{len(documents)} documents")
        if args.debug:
            documents = documents[:30]
            doc_paths = doc_ids[:30]
        kwargs = {}
        if args.query_max_length > 0:
            kwargs = {"query_max_length": args.query_max_length}
        if args.doc_max_length > 0:
            kwargs.update({"doc_max_length": args.doc_max_length})
        if args.encode_batch_size > 0:
            kwargs.update({"batch_size": args.encode_batch_size})
        if args.key is not None:
            kwargs.update({"key": args.key})
        if args.ignore_cache:
            kwargs.update({"ignore_cache": args.ignore_cache})

        # additional kwargs
        if args.generation_batch_size > 0:
            kwargs["generation_batch_size"] = args.generation_batch_size

        kwargs["overwrite"] = args.overwrite
        kwargs["no_generation"] = args.no_generation

        if reasons is not None:
            print(f"using {args.reasoning} reasonings")
            assert len(reasons) == len(queries)

            if args.model == "gem":
                kwargs["reasons"] = reasons
            else:
                queries = reasons

            kwargs["gem_instruction"] = gem_instruction

        # compatible with BRIGHT code
        instructions = None
        if config is not None:
            instructions = config["instructions_long"] if args.long_context else config["instructions"]

        scores = RETRIEVAL_FUNCS[args.model](
            queries=queries,
            query_ids=query_ids,
            documents=documents,
            excluded_ids=excluded_ids,
            instructions=instructions,
            doc_ids=doc_ids,
            task=args.task,
            cache_dir=args.cache_dir,
            long_context=args.long_context,
            model_id=args.model,
            checkpoint=args.checkpoint,
            **kwargs,
        )
        with open(score_file_path, "w") as f:
            json.dump(scores, f, indent=2)
    else:
        with open(score_file_path) as f:
            scores = json.load(f)
        print(score_file_path, "exists")
    if args.long_context:
        key = "gold_ids_long"
    else:
        key = "gold_ids"
    ground_truth = {}
    for e in tqdm(examples):
        ground_truth[e["id"]] = {}
        for gid in e[key]:
            ground_truth[e["id"]][gid] = 1
        for did in e["excluded_ids"]:
            assert not did in scores[e["id"]]
            assert not did in ground_truth[e["id"]]

    print(args.output_dir)
    results = calculate_retrieval_metrics(results=scores, qrels=ground_truth)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


def parse_args():
    """Parse arguments.

    Adapted from:
        https://github.com/xlang-ai/BRIGHT/blob/main/run.py
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=TASKS,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=_RETRIEVERS,
    )
    parser.add_argument("--long_context", action="store_true")
    parser.add_argument("--query_max_length", type=int, default=-1)
    parser.add_argument("--doc_max_length", type=int, default=-1)
    parser.add_argument("--encode_batch_size", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default=str(DATA_DIR / "results/bright"))
    parser.add_argument("--cache_dir", type=str, default=str(INDEX_DIR / "bright"))
    parser.add_argument(
        "--config_dir", type=str, default=str(ROOT_DIR / "third_party/ReasonIR/evaluation/bright/configs")
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--reasoning", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ignore_cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--generation_batch_size", type=int, default=-1)
    parser.add_argument("--no_generation", action="store_true")
    parser.add_argument("--gem_reason_file", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_bright(Args(**args.__dict__))
