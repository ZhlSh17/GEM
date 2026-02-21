import functools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset


def _adapt_format(
    item: dict,
    idx: int,
    data_source: str,
    pos_field: str = "pos",
    neg_field: str = "neg",
) -> dict:
    if isinstance(item["query"], str):
        query = item["query"]
    elif isinstance(item["query"], list) and len(item["query"]) == 2 and isinstance(item["query"][1], str):
        query = item["query"][1]
    else:
        raise NotImplementedError(f"unsupported format {item['query']}")

    ret = {"query": query, "_source": f"{data_source}_{idx}"}

    for ret_key, key in [("pos", pos_field), ("neg", neg_field)]:
        val = item[key]
        if not (isinstance(val, list) and val):
            raise RuntimeError(f"unexpected format {item[key]}")

        passages = []
        for p in val:
            if isinstance(p, str):
                passages.append(p)
            elif isinstance(p, list) and len(p) == 2 and isinstance(p[1], str):
                passages.append(p[1])
            else:
                raise NotImplementedError(f"unexpected format {item[key]}")

        ret[ret_key] = passages

    return ret


def map_format(
    dataset: "Dataset",
    data_source: str,
    *,
    pos_field: str = "pos",
    neg_field: str = "neg",
    num_proc: int | None = None,
) -> "Dataset":
    """Adapt data format.

    Returns:
        Dataset: A dataset that has the following schema:
            ```
            query: str
            pos: list[str]
            neg: list[str]
            _source: str
            ```
    """
    return dataset.map(
        functools.partial(
            _adapt_format,
            data_source=data_source,
            pos_field=pos_field,
            neg_field=neg_field,
        ),
        with_indices=True,
        batched=False,
        num_proc=num_proc,
        desc=f"reformat {data_source}",
    )
