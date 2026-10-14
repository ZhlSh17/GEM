import os
from typing import Any
from pathlib import Path

import torch
import pyterrier as pt
import pyterrier_dr as ptd
import pandas as pd
import numpy as np
from tqdm import tqdm
from ir_measures import RR, nDCG

from gem.exp.helpers import parse_known_args
from gem.model import GEM

# Copie from HyDE paper: https://aclanthology.org/2023.acl-long.99.pdf
QUERY_INSTRUCTION = """Please write a passage to answer the question."""


class GEM_PT(ptd.BiEncoder):

    def __init__(
        self,
        model: str | GEM,
        no_generation: bool = False,
        generation_args: dict | None = None,
        max_query_length: int = 1024,
        max_doc_length: int = 1024,
        query_instruction: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.model = (
            model
            if isinstance(model, GEM)
            else GEM.from_pretrained(model, dtype=torch.bfloat16, device_map="auto").eval()
        )

        self.no_generation = no_generation
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length

        self.generation_args = generation_args or {}
        self.generation_args.setdefault("do_sample", False)
        self.generation_args.setdefault("max_new_tokens", 1024)

        self.query_instruction = query_instruction

    def encode_queries(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        verbose = self.verbose
        if isinstance(texts, tqdm):
            texts.disable = True
            verbose = True

        texts = list(texts)
        if len(texts) == 0:
            return np.empty(shape=(0, 0))

        if self.no_generation:
            embs = self.model.encode_corpus(
                texts,
                batch_size=batch_size,
                show_progress_bar=verbose,
                max_length=self.max_query_length,
            )

        else:
            embs = self.model.encode_queries(
                texts,
                batch_size=batch_size,
                show_progress_bar=verbose,
                max_length=self.max_query_length,
                instruction=self.query_instruction,
                return_generation=False,
                **self.generation_args,
            )

        return embs.cpu().float().numpy()

    def encode_docs(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        verbose = self.verbose
        if isinstance(texts, tqdm):
            texts.disable = True
            verbose = True

        texts = list(texts)
        if len(texts) == 0:
            return np.empty(shape=(0, 0))

        embs = self.model.encode_corpus(
            texts,
            batch_size=batch_size,
            show_progress_bar=verbose,
            max_length=self.max_doc_length,
        )

        return embs.cpu().float().numpy()


def main():
    args, _ = parse_known_args()

    model = GEM_PT(
        args.checkpoint,
        no_generation=args.no_generation,
        query_instruction=QUERY_INSTRUCTION,  # for generate-then-encode
    )
    index = ptd.FlexIndex(args.index_dir)

    if not os.listdir(args.index_dir):
        pipeline: pt.Compose = model.doc_encoder(batch_size=args.batch_size) >> index.indexer()
        pipeline.index(
            pt.get_dataset("irds:msmarco-passage").get_corpus_iter(),
            batch_size=args.batch_size,
        )

    retr = model.query_encoder(
        batch_size=args.generation_batch_size or args.batch_size,
        verbose=True,
    ) >> index.np_retriever(num_results=100)

    for dataset_name in [
        "irds:msmarco-passage/trec-dl-2020/judged",
        "irds:msmarco-passage/trec-dl-2019/judged",
    ]:
        dataset = pt.get_dataset(dataset_name)
        save_dir = Path(args.output_dir) / dataset_name.split("/")[1]
        save_dir.mkdir(parents=True, exist_ok=True)

        df: pd.DataFrame = pt.Experiment(
            [retr],
            topics=dataset.get_topics(),
            qrels=dataset.get_qrels(),
            eval_metrics=[RR @ 10, nDCG @ 10],
            names=["GEM"],
            save_dir=save_dir,
        )
        df.to_csv(save_dir / "metrics.csv")

        print(dataset_name)
        print(df)


if __name__ == "__main__":
    main()
