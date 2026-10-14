import os
import json
import functools
import argparse
from typing import Any, cast
from collections.abc import Callable

import numpy as np
import torch
import mteb
from mteb.evaluation.evaluators.RetrievalEvaluator import DRESModel
from loguru import logger

from gem.model import GEM
from gem.exp import BiEncoder
from gem.exp.repllama import Repllama
from gem.exp.promptriever import Promptriever
from gem.exp.helpers import parse_known_args, BaseEncoder, GEMEncoder

TASK_NAMES = [
    "Core17InstructionRetrieval",
    "News21InstructionRetrieval",
    "Robust04InstructionRetrieval",
]


def preprocess_queries(queries: list[str], instructions: dict[str, str]) -> list[str]:
    if instructions:
        queries = [f"{query} {instructions[query]}".strip() for query in queries]

    return queries


# Adapted from: https://github.com/orionw/FollowIR/blob/master/models/e5/evaluate_e5.py
class MTEBEncoder(DRESModel):
    """Wrapper for MTEB.

    See Also:
        :class:`mteb.evaluation.evaluators.RetrievalEvaluator.DRESModel`

    """

    def __init__(
        self,
        model: BiEncoder,
        verbose: bool = True,
        save_corpus_embeddings: bool = True,
        sep: str = "\n",
        **kwargs: Any,
    ) -> None:
        super().__init__(model, save_corpus_embeddings=save_corpus_embeddings, **kwargs)

        self.verbose = verbose
        self.sep = sep
        self.model = BaseEncoder(model)

    def encode_queries(self, queries: list[str], batch_size: int, **kwargs: Any) -> np.ndarray:
        if self.verbose:
            logger.debug(f"{queries[0]=}")

        instructions = kwargs.pop("instructions")
        preprocess_queries(queries, instructions)

        if self.verbose and instructions:
            logger.debug(f"{queries[0]=}")

        return self.model.encode_queries(
            queries,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=False,
        )

    def encode_corpus(self, corpus: list[dict[str, str]], batch_size: int, **kwargs: Any) -> np.ndarray:
        qid = kwargs.get("qid")
        if qid and self.save_corpus_embeddings and qid in self.corpus_embeddings:
            return self.corpus_embeddings[qid]

        assert isinstance(corpus, list) and isinstance(corpus[0], dict)

        texts = []
        for doc in corpus:
            title = doc.get("title")
            text = doc["text"]
            texts.append(f"{title}{self.sep}{text}" if title else text)

        embs = self.model.encode_corpus(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=False,
        )

        if self.save_corpus_embeddings and qid:
            self.corpus_embeddings[qid] = embs

        return embs


class GEM_MTEB(MTEBEncoder):

    def __init__(
        self,
        model: GEM,
        *,
        verbose: bool = True,
        no_generation: bool = False,
        generation_batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, verbose=verbose, **kwargs)

        self.model = GEMEncoder(
            model,
            no_generation=no_generation,
            generation_batch_size=generation_batch_size,
        )


def encode_query_expansion(
    encode_queries: Callable,
    qe_file: str,
    is_model_gem: bool = False,
) -> Callable:
    logger.info(f"using pre-computed query expansion file: {qe_file}")
    with open(qe_file, "rt") as f:
        query2expansion: dict[str, str] = json.load(f)

    @functools.wraps(encode_queries)
    def wrapper_query2doc(queries: list[str], *args, **kwargs) -> torch.Tensor:
        # Following Query2Doc (https://aclanthology.org/2023.emnlp-main.585.pdf), we concatenate query and QE
        # Minor difference: "[SEP]" is not for modern LLM-based embedding model, we use "\n\n" as the separator instead.
        for q in queries:
            assert q in query2expansion

        queries = [f"{q}\n\n{query2expansion[q]}" for q in queries]
        return encode_queries(queries, *args, **kwargs)

    @functools.wraps(encode_queries)
    def wrapper_hyde(queries: list[str], *args, **kwargs) -> torch.Tensor:
        for q in queries:
            assert q in query2expansion

        q_embs = encode_queries(queries, *args, **kwargs)

        # only one QE per query for fair comparison
        queries = [query2expansion[q] for q in queries]
        qe_embs = encode_queries(queries, *args, **kwargs)

        return (q_embs + qe_embs) / 2

    @functools.wraps(encode_queries)
    def wrapper_gem(queries: list[str], *args, **kwargs) -> torch.Tensor:
        for q in queries:
            assert q in query2expansion

        # simply concatenate query and precomputed GEM's response; same as query2doc
        queries = [f"{q}\n\n{query2expansion[q]}" for q in queries]
        return encode_queries(queries, *args, **kwargs)

    # for our ablation study
    @functools.wraps(encode_queries)
    def wrapper_add_gem_reason(queries: list[str], *args, **kwargs) -> torch.Tensor:
        for q in queries:
            assert q in query2expansion

        reasons = [query2expansion[q] for q in queries]

        return encode_queries(queries, *args, reason=reasons, **kwargs)

    if "query2doc" in qe_file:
        return wrapper_query2doc
    elif "hyde" in qe_file:
        return wrapper_hyde
    elif "gem" in qe_file:
        return wrapper_add_gem_reason if is_model_gem else wrapper_gem
    else:
        raise NotImplementedError(f"unsupported QE method: {qe_file}")


# Adapted from official code: https://github.com/orionw/FollowIR/blob/master/scripts/gather_results.py#L48
def log_results(output_dir: str, task_names: list[str] | None = None) -> None:
    if task_names is None:
        task_names = TASK_NAMES

    all_main_score = []
    all_p_mrr = []

    for task in task_names:
        path = os.path.join(output_dir, f"{task}.json")
        if not os.path.exists(path):
            continue

        with open(path, "rt") as f:
            data = json.load(f)["test"]

        rankwise = data["p-MRR"]

        # map@1000 and ndcg@5
        map1000 = data["individual"]["original"]["map_at_1000"]
        ndcg5 = data["individual"]["original"]["ndcg_at_5"]

        # map@1000 and ndcg@5 of the changed
        map1000_changed = data["individual"]["changed"]["map_at_1000"]
        ndcg5_changed = data["individual"]["changed"]["ndcg_at_5"]

        # map@100 and ndcg@5 of the base
        map1000_base = data["length_ablation"]["base"]["map_at_1000"]
        ndcg5_base = data["length_ablation"]["base"]["ndcg_at_5"]
        diff_map1000 = map1000 - map1000_base
        diff_ndcg5 = ndcg5 - ndcg5_base

        main_score = map1000 if "news" not in task.lower() else ndcg5
        results = {
            "dataset": task,
            "map": map1000,
            "ndcg@5": ndcg5,
            "map_changed": map1000_changed,
            "ndcg@5_changed": ndcg5_changed,
            "map_base": map1000_base,
            "ndcg@5_base": ndcg5_base,
            "diff_score": diff_map1000 if "news" not in task.lower() else diff_ndcg5,
            "main_score": main_score,
            "p-MRR": rankwise,
        }

        for key in results:
            if isinstance(results[key], float):
                results[key] *= 100

        all_main_score.append(results["main_score"])
        all_p_mrr.append(results["p-MRR"])

        logger.info(f"{task=} results:\n{json.dumps(results, indent=4)}")

    if all_main_score and all_p_mrr:
        logger.info(
            f"\nAverage score: {sum(all_main_score) / len(all_main_score)}\nAverage p-MRR: {sum(all_p_mrr) / len(all_p_mrr)}"
        )


def load_mteb_model(args: argparse.Namespace) -> MTEBEncoder:
    model = None

    if args.model == "gem":
        if args.checkpoint is None:
            raise ValueError("require GEM checkpoint")

        model = GEM_MTEB(
            GEM.from_pretrained(args.checkpoint, device_map="auto", dtype=torch.bfloat16).eval(),
            no_generation=args.no_generation,
            verbose=args.verbose,
            generation_batch_size=args.generation_batch_size,
        )

    elif args.model == "repllama":
        # sep=" " for consistency with official example: https://huggingface.co/castorini/repllama-v1-7b-lora-passage
        model = MTEBEncoder(Repllama(), verbose=args.verbose, sep=" ")

    elif args.model == "promptriever":
        model = MTEBEncoder(Promptriever(), verbose=args.verbose, sep=" ")

    elif args.model == "reasonir":
        from gem.exp.reasonir import ReasonIR

        model = MTEBEncoder(ReasonIR(), verbose=args.verbose)

    elif args.model == "bge":
        from gem.exp.bge import BGE

        model = MTEBEncoder(BGE(), verbose=args.verbose)

    else:
        raise NotImplementedError(f"unsupported model: {args.model}")

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qe_file", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args, extra = parse_known_args()
    return parser.parse_args(extra, args)


# Adapted from: https://github.com/orionw/FollowIR/blob/master/models/e5/evaluate_e5.py
def main():
    args = parse_args()
    batch_size = args.batch_size
    output_dir = args.output_dir

    # FollowIR
    task_names = TASK_NAMES

    if os.path.exists(output_dir) and os.listdir(output_dir):
        log_results(output_dir, task_names)
        raise FileExistsError(f"{output_dir} exists")

    logger.info(f"{task_names=}")

    model = load_mteb_model(args)
    if args.qe_file is not None:
        model.model.encode_queries = encode_query_expansion(
            model.model.encode_queries,
            args.qe_file,
            is_model_gem=args.model == "gem",
        )

    for task in task_names:
        evaluation = mteb.MTEB(tasks=[task], task_langs=["en"], do_length_ablation=True)
        evaluation.run(
            model,
            eval_splits=["test"],
            output_folder=output_dir,
            save_corpus_embeddings=True,
            batch_size=batch_size,
        )

    log_results(output_dir, task_names)


if __name__ == "__main__":
    main()
