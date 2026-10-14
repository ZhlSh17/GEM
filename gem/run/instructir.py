import os
import json
import random
import functools
from typing import Any
from pathlib import Path

from loguru import logger
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from third_party.InstructIR.eval.robust_eval import CustomEvaluateRetrieval
from datasets import load_dataset as load_dataset_hf

from gem.utils import HF_DATASETS_DIR
from gem.exp.helpers import load_model, parse_known_args


def _preproc_query(text: str, inverse: bool = False) -> str:
    if inverse:
        return "[SEP]".join(text.rsplit("Question:", maxsplit=1))
    else:
        return text.replace("[SEP]", "Question:")


def load_instructir() -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
    _load_dataset_hf = functools.partial(load_dataset_hf, "kaist-ai/InstructIR", cache_dir=HF_DATASETS_DIR)
    queries: list[dict[str, str]] = _load_dataset_hf("queries")["queries"].to_list()
    corpus: list[dict[str, str]] = _load_dataset_hf("corpus")["corpus"].to_list()
    qrels: list[dict[str, str]] = _load_dataset_hf("qrels")["test"].to_list()

    queries = {x["_id"]: _preproc_query(x["text"]) for x in queries}
    corpus = {x["_id"]: {"title": x.get("title", ""), "text": x.get("text", "")} for x in corpus}

    _qrels = {}
    for x in qrels:
        qid = x["query-id"]
        docid = x["corpus-id"]
        score = int(x["score"])
        if qid not in _qrels:
            _qrels[qid] = {docid: score}
        else:
            _qrels[qid][docid] = score

    return corpus, queries, _qrels


def log_example_ranking(
    corpus: dict[str, dict[str, Any]],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    top_k: int = 10,
) -> None:
    if not results:
        return

    random.seed(42)
    query_id = random.choice(list(results))
    ranking_scores = results[query_id]
    scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)

    logger.info(f"sample qid: {query_id}")
    logger.info(f"sample query: {queries[query_id]}")

    target_pid = next(iter(qrels[query_id]))
    logger.info(f"gold pid: {target_pid}")
    logger.info(f"gold text: {corpus[target_pid]['text']}")

    for rank, (doc_id, _) in enumerate(scores_sorted[:top_k], start=1):
        rel = qrels[query_id].get(doc_id, 0)
        title = corpus[doc_id].get("title")
        logger.info(f"rank {rank}: rel={rel} doc_id={doc_id} title={title!r}")
        logger.info(corpus[doc_id]["text"])


def log_results(output_dir: str | os.PathLike | None = None, metrics: dict | None = None) -> None:
    if metrics is None:
        path = os.path.join(output_dir, "metrics.json")
        if os.path.exists(path):
            with open(path, "rt") as f:
                metrics = json.load(f)

    logger.info(f"\nRobustness@10: {metrics['cluster_NDCG@10_min_avg']}\nnDCG@10: {metrics['cluster_NDCG@10_avg']}")


# Adapted from: https://github.com/kaistAI/InstructIR/blob/main/eval/eval_instructir_dense.py
def main():
    args = parse_known_args()[0]
    corpus, queries, qrels = load_instructir()
    logger.info(f"loaded corpus={len(corpus)} queries={len(queries)} qrels={len(qrels)}")

    retriever = CustomEvaluateRetrieval(DRES(load_model(args), batch_size=args.batch_size))
    results = retriever.retrieve(corpus, queries)

    queries = {qid: _preproc_query(text, inverse=True) for qid, text in queries.items()}
    metrics = retriever.robustness_evaluate(
        queries=queries,
        qrels=qrels,
        results=results,
        k_values=retriever.k_values,
        type="query",
    )

    log_results(metrics=metrics)

    if args.output_dir:
        save_path = Path(args.output_dir) / "metrics.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wt") as f:
            json.dump(metrics, f)

        logger.info(f"saved metrics to {save_path}")


if __name__ == "__main__":
    main()
