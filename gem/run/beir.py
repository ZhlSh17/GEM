import argparse
from typing import Any
from pathlib import Path

import torch
import numpy as np
from loguru import logger
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.models.util import extract_corpus_sentences

from gem.exp import BiEncoder
from gem.model import GEM
from gem.utils import DATA_DIR


def parse_known_args() -> tuple[argparse.Namespace, list]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gem")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--generation_batch_size", type=int, default=None)
    parser.add_argument("--no_generation", action="store_true")
    parser.add_argument("--index_dir", type=str, default=None)
    # parser.add_argument("--overwrite", action="store_true")
    args, extra = parser.parse_known_args()
    return args, extra


class BiEncoder_BEIR:

    def __init__(self, model: BiEncoder, sep: str = "\n", **kwargs: Any) -> None:
        self.model = model
        self.sep = sep

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor | np.ndarray:
        embs = self.model.encode_query(queries, batch_size=batch_size, verbose=show_progress_bar, **kwargs)
        if convert_to_tensor:
            return embs.float()  # numpy doesn't support bfloat16

        return embs.cpu().float().numpy()

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, list] | list[str],
        batch_size: int,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor | np.ndarray:
        texts = extract_corpus_sentences(corpus=corpus, sep=self.sep)
        embs = self.model.encode_passage(texts, batch_size=batch_size, verbose=show_progress_bar, **kwargs)
        if convert_to_tensor:
            return embs.float()

        return embs.cpu().float().numpy()


class GEM_BEIR(BiEncoder_BEIR):

    def __init__(
        self,
        model: GEM,
        *,
        no_generation: bool = False,
        generation_batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)

        self.no_generation = no_generation
        self.generation_batch_size = generation_batch_size

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = True,
        *,
        reasons: list[str] | None = None,
        max_length: int | None = 1024,
        **kwargs: Any,
    ) -> torch.Tensor | np.ndarray:
        model: GEM = self.model
        if self.no_generation:
            embs = model.encode_passage(
                queries,
                batch_size=batch_size,
                verbose=show_progress_bar,
                max_length=max_length,
            )
        else:
            embs = model.encode_query(
                queries,
                batch_size=self.generation_batch_size or batch_size,
                verbose=show_progress_bar,
                max_length=max_length,
                max_new_tokens=kwargs.pop("max_new_tokens", 1024),
                do_sample=kwargs.pop("do_sample", False),
                reason=reasons,
                **kwargs,
            )

        if convert_to_tensor:
            return embs.float()

        return embs.cpu().float().numpy()

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, list] | list[str],
        batch_size: int,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = True,
        *,
        max_length: int | None = 1024,
        **kwargs: Any,
    ) -> torch.Tensor | np.ndarray:
        return super().encode_corpus(
            corpus,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=convert_to_tensor,
            max_length=max_length,
            **kwargs,
        )


def load_beir_model(args: argparse.Namespace) -> BiEncoder_BEIR:
    model = None

    if args.model == "gem":
        if args.checkpoint is None:
            raise ValueError("require GEM checkpoint")

        model = GEM_BEIR(
            GEM.from_pretrained(args.checkpoint, device_map="auto", dtype=torch.bfloat16).eval(),
            no_generation=args.no_generation,
            generation_batch_size=args.generation_batch_size,
        )

    elif args.model == "repllama":
        from gem.exp.repllama import Repllama

        # sep=" " just to be consistent with official example: https://huggingface.co/castorini/repllama-v1-7b-lora-passage
        model = BiEncoder_BEIR(Repllama(), sep=" ")

    elif args.model == "promptriever":
        from gem.exp.promptriever import Promptriever

        model = BiEncoder_BEIR(Promptriever(), sep=" ")

    elif args.model == "reasonir":
        from gem.exp.reasonir import ReasonIR

        model = BiEncoder_BEIR(ReasonIR())

    else:
        raise NotImplementedError(f"unsupported model: {args.model}")

    return model


# Adapted from: https://github.com/beir-cellar/beir#beers-quick-example
def eval_beir(
    retriever: EvaluateRetrieval,
    dataset: str,
    args: argparse.Namespace,
    split: str = "test",
    overwrite_query_embs: bool = False,
) -> None:
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = beir_util.download_and_unzip(url, DATA_DIR / "datasets")

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split)

    index_dir = Path(args.index_dir) / dataset
    index_dir.mkdir(parents=True, exist_ok=True)

    q_embs_path = index_dir / "queries.pkl"
    if q_embs_path.exists() and overwrite_query_embs:
        logger.info("will overwrite query embeddings")
        q_embs_path.unlink()

    logger.info(f"encoding {dataset}/{split}")
    results = retriever.encode_and_retrieve(
        corpus=corpus,
        queries=queries,
        encode_output_path=index_dir,
        overwrite=False,  # this will overwrite both query and doc embeddings
    )

    ndcg, _map, recall, precision = retriever.evaluate(qrels=qrels, results=results, k_values=retriever.k_values)
    mrr = retriever.evaluate_custom(qrels=qrels, results=results, k_values=retriever.k_values, metric="mrr")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    beir_util.save_runfile(output_dir / f"{dataset}-{split}.run.trec", results)
    beir_util.save_results(
        output_dir / f"{dataset}-{split}.json",
        ndcg=ndcg,
        _map=_map,
        recall=recall,
        precision=precision,
        mrr=mrr,
    )


def _make_trec_dl20_patch() -> None:
    import ir_datasets
    import pandas as pd
    from gem.utils import load_jsonl, save_jsonl

    dataset = "msmarco"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = beir_util.download_and_unzip(url, DATA_DIR / "datasets")
    data_path = Path(data_path)

    if (data_path / "qrels/trec_dl20.tsv").exists():
        logger.info("trec-dl20 patch already applied")
        return

    _dataset = ir_datasets.load("msmarco-passage/trec-dl-2020/judged")

    qrels = pd.DataFrame(_dataset.qrels_iter())
    qrels: pd.DataFrame = qrels.rename(
        columns={"query_id": "query-id", "doc_id": "corpus-id", "relevance": "score"}
    ).drop(columns="iteration")
    qrels.to_csv(data_path / "qrels/trec_dl20.tsv", sep="\t", index=False)

    queries = load_jsonl(data_path / "queries.jsonl")

    new_queries = []
    for query in _dataset.queries_iter():
        new_queries.append({"_id": query.query_id, "text": query.text, "metadata": {}})

    save_jsonl(queries + new_queries, data_path / "queries.jsonl")


def main():
    args, _ = parse_known_args()

    retriever = EvaluateRetrieval(DRES(load_beir_model(args), batch_size=args.batch_size))

    # _make_trec_dl20_patch()
    for split in ["dev", "test", "trec_dl20"]:
        eval_beir(
            retriever=retriever,
            dataset="msmarco",
            args=args,
            split=split,
            overwrite_query_embs=True,
        )


if __name__ == "__main__":
    main()
