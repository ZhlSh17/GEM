import os
from typing import Any, Literal

from datasets import load_dataset as _load_dataset_hf, Dataset, DatasetDict

from gem.utils import HF_DATASETS_DIR
from gem.datasets.format import map_format


def load_raw_dataset(
    subset: str,
    *,
    cache_dir: str | os.PathLike = HF_DATASETS_DIR,
    map_id_to_doc: bool = False,
    **kwargs: Any,
) -> DatasetDict:
    dataset = _load_dataset_hf("reasonir/reasonir-data", subset, cache_dir=cache_dir, **kwargs)

    if map_id_to_doc and subset == "hq":
        # Subset "hq" comes with doc ids instead of doc content
        dataset = _map_id_to_doc(dataset)

    return dataset


def load_dataset(
    subset: Literal["hq", "vl"],
    cache_dir: str | os.PathLike = HF_DATASETS_DIR,
    num_proc: int | None = None,
    **kwargs: Any,
) -> Dataset:
    return map_format(
        dataset=load_raw_dataset(subset, cache_dir=cache_dir, map_id_to_doc=True, **kwargs)["train"],
        data_source=f"reasonir_{subset}",
        num_proc=num_proc,
    )


def _map_id_to_doc(hq_dataset: Dataset) -> Dataset:
    # Adapted from: https://huggingface.co/datasets/reasonir/reasonir-data#hard-query-hq-data
    from gem.datasets.bright import load_dataset as load_bright_dataset

    def get_doc_and_ids(doc_pairs):
        doc_ids = []
        documents = []
        for dp in doc_pairs:
            doc_ids.append(str(dp["id"]))
            documents.append(dp["content"])
        return documents, doc_ids

    def process_pos_id2doc(entry, id2doc):
        pos_docs = entry["pos"]
        res = []
        for pos in pos_docs:
            instruction, doc_id = pos[0], pos[1]
            doc = id2doc[doc_id]
            res.append([instruction, doc])
        entry["pos"] = res
        return entry

    bright_docs = load_bright_dataset("documents")
    all_docs = []
    all_ids = []
    for task in bright_docs.keys():
        docs, ids = get_doc_and_ids(bright_docs[task])
        all_docs.extend(docs)
        all_ids.extend(ids)

    id2doc = {}
    for i in range(len(all_docs)):
        id2doc[all_ids[i]] = all_docs[i]

    return hq_dataset.map(
        lambda x: process_pos_id2doc(x, id2doc),
        desc="map doc ids to doc content for reasonir/hq",
    )
