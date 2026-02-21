import json
import math
import random
import functools
from dataclasses import dataclass
from collections import defaultdict
from typing import Any

import torch
from tqdm import tqdm
from loguru import logger
from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizerBase, BatchEncoding
from transformers.data.data_collator import DataCollatorWithPadding, pad_without_fast_tokenizer_warning

from gem.preprocess import Preprocessor, ensure_emb_token_is_added


class GEMDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset: HFDataset,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_query_len: int = 4096,
        max_passage_len: int = 4096,
        train_group_size: int = 2,
        num_proc: int = 4,
        embed_only: bool = False,
    ) -> None:
        self.max_query_len = max_query_len
        self.max_passage_len = max_passage_len
        self.num_proc = num_proc
        self.embed_only = embed_only
        if self.embed_only:
            logger.info("create dataset for embedding-only ablation")

        if train_group_size < 1:
            raise ValueError(f"{train_group_size=}; expect a value >= 1")
        self.train_group_size = train_group_size

        self._dataset = dataset

        ensure_emb_token_is_added(tokenizer)
        self.tokenizer = tokenizer

        self._is_preprocess_done = False

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict:
        if not self._is_preprocess_done:
            logger.info("lazy preprocess triggered")
            self.preprocess()

        item = self._dataset[idx]

        pos = random.choice(item["pos"])
        num_negs = self.train_group_size - 1
        all_negs = item["neg"]
        if len(all_negs) < num_negs:
            negs = random.choices(all_negs, k=num_negs)
        else:
            negs = random.sample(all_negs, k=num_negs)

        return {
            "query": item["query"],
            "passage": [pos] + negs,
        }

    def preprocess(self) -> "GEMDataset":
        if self._is_preprocess_done:
            return self

        self._pretokenize_dataset()
        self._filter_dataset()

        self._is_preprocess_done = True
        return self

    def _pretokenize_dataset(self) -> None:
        for field in ["query", "pos", "neg"]:
            if field not in self._dataset.features:
                raise RuntimeError(f"{field=} not exists in {self._dataset}")

        remove_cols = []
        for col in ["instruction", "reason"]:
            if col in self._dataset.features:
                remove_cols.append(col)

        self._dataset = self._dataset.map(
            functools.partial(
                self._elementwise_tokenize,
                preproc=Preprocessor(self.tokenizer),
                append_embed_token=True,
                embed_only=self.embed_only,
                max_passage_len=self.max_passage_len,
            ),
            remove_columns=remove_cols,
            batched=False,
            desc="pretokenize",
            num_proc=self.num_proc,
        )

        logger.info("pretokenization done")

    def _filter_dataset(self) -> None:
        size = len(self._dataset)

        logger.info(f"start filtering by {self.max_query_len=}")
        self._count_source(self._dataset)
        self._dataset = self._dataset.filter(
            functools.partial(self._filter, max_query_len=self.max_query_len),
            batched=False,
            num_proc=self.num_proc,
        )

        if size > len(self._dataset):
            logger.info(f"dataset size {size} -> {len(self._dataset)} after filtering")
            self._count_source(self._dataset)

    def _count_source(self, dataset: HFDataset) -> None:
        if "_source" not in dataset.features:
            return

        source_counts = {"data_with_reason": defaultdict(int), "data_without_reason": defaultdict(int)}

        batch_size = 1024
        for batch in tqdm(
            dataset.remove_columns(["query", "pos", "neg"]).iter(batch_size),
            total=math.ceil(len(dataset) / batch_size),
            desc="compute data source stats",
        ):
            for idx in range(len(batch["_source"])):
                _source_counts = source_counts["data_without_reason"]

                if isinstance(batch.get("has_reason"), list) and batch["has_reason"][idx]:
                    _source_counts = source_counts["data_with_reason"]

                source = batch["_source"][idx]
                if not isinstance(source, str):
                    _source_counts[None] += 1
                    continue

                source = source.rsplit("_", maxsplit=1)[0]
                _source_counts[source] += 1

        logger.info(f"Number of samples per data source:\n{json.dumps(source_counts, indent=2)}\n")

    @staticmethod
    def _elementwise_tokenize(
        item: dict,
        *,
        preproc: Preprocessor,
        append_embed_token: bool = True,
        embed_only: bool = False,
        max_passage_len: int = 4096,
    ) -> dict:
        def truncate(text: str, max_length: int) -> str:
            return preproc.tokenizer.decode(
                preproc.tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
            )

        results = {"_source": item.get("_source")}

        # tokenize query
        if embed_only or item.get("reason") is None:
            results["has_reason"] = False
            text = preproc.preprocess_passage(passage=item["query"], append_embed_token=append_embed_token)
            results["query"] = preproc.tokenizer(text, add_special_tokens=False)
            # no generation loss in embed-only ablation
            results["query"]["labels"] = [-100] * len(results["query"]["input_ids"])
        else:
            results["has_reason"] = True
            text = preproc.preprocess_query(
                query=item["query"],
                reason=item["reason"],
                instruction=item["instruction"],
                append_embed_token=append_embed_token,
            )

            results["query"] = preproc.tokenizer(text, add_special_tokens=False)

            prompt = preproc.preprocess_query(
                query=item["query"],
                reason=None,
                instruction=item["instruction"],
                append_embed_token=False,
            )

            prompt_len = len(preproc.tokenizer(prompt, add_special_tokens=False)["input_ids"])

            # mask out prompt tokens for next-token prediction
            labels = [-100 if idx < prompt_len else tok_id for idx, tok_id in enumerate(results["query"]["input_ids"])]

            if append_embed_token:
                labels[-1] = -100

            results["query"]["labels"] = labels

        # tokenize passages
        for key in ["pos", "neg"]:
            val = []
            for passage in item[key]:
                text = preproc.preprocess_passage(
                    passage=truncate(passage, max_passage_len), append_embed_token=append_embed_token
                )
                val.append(preproc.tokenizer(text, add_special_tokens=False))

            results[key] = val

        return results

    @staticmethod
    def _filter(
        item: dict,
        *,
        max_query_len: int,
    ) -> bool:
        if len(item["query"]["input_ids"]) > max_query_len:
            return False

        return item["pos"] and item["neg"]


@dataclass
class GEMDataCollator(DataCollatorWithPadding):

    def __post_init__(self) -> None:
        if self.tokenizer.padding_side != "left":
            logger.info("set tokenizer padding_side to `left`")
            self.tokenizer.padding_side = "left"

    def __call__(self, samples: list[dict]) -> dict[str, Any]:
        queries: list[dict[str, Any]] = []
        passages: list[dict[str, Any]] = []

        for item in samples:
            queries.append(item["query"])
            passages.extend(item["passage"])

        # query batch
        q_batch: BatchEncoding = pad_without_fast_tokenizer_warning(self.tokenizer, queries, return_tensors=None)
        labels_batch = []
        for input_ids, labels in zip(q_batch["input_ids"], q_batch["labels"]):
            num_pad_tokens = len(input_ids) - len(labels)
            if num_pad_tokens > 0:
                labels = [-100] * num_pad_tokens + labels
            labels_batch.append(labels)
        q_batch["labels"] = labels_batch
        q_batch = q_batch.convert_to_tensors(self.return_tensors)

        # passage batch
        p_batch = pad_without_fast_tokenizer_warning(self.tokenizer, passages, return_tensors=self.return_tensors)

        return {
            "query": q_batch,
            "passage": p_batch,
            "labels": q_batch.pop("labels"),
        }
