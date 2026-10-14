import os
from typing import Any

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange


# See: https://github.com/xlang-ai/BRIGHT/blob/main/retrievers.py
@torch.no_grad()
def retrieval_promptriever(
    queries: list[str],
    query_ids: list[str],
    documents: list[str],
    doc_ids: list[str],
    task: str,
    model_id: str,
    instructions: dict[str, str] | None,
    cache_dir: str,
    excluded_ids: dict,
    long_context: bool,
    **kwargs: Any,
):
    from third_party.ReasonIR.evaluation.bright.retrievers import get_scores
    from gem.exp.promptriever import Promptriever

    model = Promptriever()

    # more config
    overwrite = kwargs.get("overwrite", False)
    batch_size = kwargs.get("batch_size", 1)
    doc_max_length = kwargs.get("doc_max_length", 1024)
    query_max_length = kwargs.get("query_max_length", 2048)
    print(f"{query_max_length=}")
    print(f"{doc_max_length=}")

    # The following code is adapted from https://github.com/xlang-ai/BRIGHT/blob/main/retrievers.py#L154
    doc_emb = None
    cache_path = os.path.join(cache_dir, "doc_emb", model_id, task, f"long_{long_context}.npy")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.isfile(cache_path):
        if overwrite:
            # remove the file because `np.save` is incremental
            os.remove(cache_path)
        else:
            # already exists so we can just load it
            doc_emb = np.load(cache_path, allow_pickle=True)

    if not (isinstance(doc_emb, np.ndarray) and len(doc_emb) == len(documents)):
        for start_idx in trange(0, len(documents), batch_size):
            assert doc_emb is None or doc_emb.shape[0] % batch_size == 0, f"{doc_emb % batch_size} reminder in doc_emb"
            if doc_emb is not None and doc_emb.shape[0] // batch_size > start_idx:
                continue

            embeddings = model.encode_corpus(
                documents[start_idx : start_idx + batch_size],
                batch_size=batch_size,
                show_progress_bar=False,
                max_length=doc_max_length,
            )

            embeddings = embeddings.cpu().float().numpy()
            doc_emb = embeddings if doc_emb is None else np.concatenate((doc_emb, embeddings), axis=0)

            # save the embeddings every 1000 iters, you can adjust this as needed
            if (start_idx + 1) % 1000 == 0:
                np.save(cache_path, doc_emb)

        np.save(cache_path, doc_emb)

    doc_emb = torch.tensor(doc_emb)
    print("doc_emb shape:", doc_emb.shape)
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    query_emb = []
    for start_idx in trange(0, len(queries), batch_size):
        embeddings = model.encode_queries(
            queries[start_idx : start_idx + batch_size],
            batch_size=batch_size,
            show_progress_bar=False,
            max_length=query_max_length,
        )

        embeddings = embeddings.cpu().float().numpy().tolist()
        query_emb += embeddings

    query_emb = torch.tensor(query_emb)
    print("query_emb shape:", query_emb.shape)
    query_emb = F.normalize(query_emb, p=2, dim=1)
    scores = (query_emb @ doc_emb.T) * 100
    scores = scores.tolist()
    return get_scores(query_ids=query_ids, doc_ids=doc_ids, scores=scores, excluded_ids=excluded_ids)


@torch.no_grad()
def retrieval_gem(
    queries: list[str],
    query_ids: list[str],
    documents: list[str],
    doc_ids: list[str],
    task: str,
    model_id: str,
    instructions: dict[str, str] | None,
    cache_dir: str,
    excluded_ids: dict,
    long_context: bool,
    **kwargs: Any,
) -> dict:
    from third_party.ReasonIR.evaluation.bright.retrievers import get_scores
    from gem.model import GEM
    from gem.exp.helpers import GEMEncoder

    checkpoint = kwargs.get("checkpoint", None)
    if checkpoint is None:
        raise ValueError("require path to checkpoint")

    if isinstance(checkpoint, GEM):
        model = checkpoint
    else:
        model = GEM.from_pretrained(checkpoint, device_map="auto", dtype=torch.bfloat16).eval()

    overwrite = kwargs.get("overwrite", False)
    batch_size = kwargs.get("batch_size", 1)
    query_max_length = kwargs.get("query_max_length", 1024)  # just for query, not generation
    doc_max_length = kwargs.get("doc_max_length", 1024)
    print(f"{query_max_length=}")
    print(f"{doc_max_length=}")

    wrapper = GEMEncoder(
        model,
        no_generation=kwargs.get("no_generation", False),
        generation_batch_size=kwargs.get("generation_batch_size") or batch_size,
    )

    # The following code is adapted from https://github.com/xlang-ai/BRIGHT/blob/main/retrievers.py#L154
    doc_emb = None
    cache_path = os.path.join(cache_dir, "doc_emb", model_id, task, f"long_{long_context}.npy")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.isfile(cache_path):
        if overwrite:
            # remove the file because `np.save` is incremental
            os.remove(cache_path)
        else:
            # already exists so we can just load it
            doc_emb = np.load(cache_path, allow_pickle=True)
    if not (isinstance(doc_emb, np.ndarray) and len(doc_emb) == len(documents)):
        for start_idx in trange(0, len(documents), batch_size, desc="encode docs"):
            assert doc_emb is None or doc_emb.shape[0] % batch_size == 0, f"{doc_emb % batch_size} reminder in doc_emb"
            if doc_emb is not None and doc_emb.shape[0] // batch_size > start_idx:
                continue

            embeddings = wrapper.encode_corpus(
                documents[start_idx : start_idx + batch_size],
                show_progress_bar=False,
                batch_size=batch_size,
                max_length=doc_max_length,
            )
            embeddings = embeddings.cpu().float().numpy()
            doc_emb = embeddings if doc_emb is None else np.concatenate((doc_emb, embeddings), axis=0)

            # save the embeddings every 1000 iters, you can adjust this as needed
            if (start_idx + 1) % 1000 == 0:
                np.save(cache_path, doc_emb)

        np.save(cache_path, doc_emb)

    doc_emb = torch.tensor(doc_emb)
    print("doc_emb shape:", doc_emb.shape)
    query_emb = []
    reasons = kwargs.get("reasons", None)
    print(f"{len(queries)=}")
    if reasons is not None:
        print(f"{len(reasons)=}")
        if kwargs.get("no_generation", False):
            queries = reasons

    instruction = kwargs.get("gem_instruction", None)
    if instruction:
        print(f"using provided instruction: {instruction}")

    batch_size = wrapper.generation_batch_size
    for start_idx in trange(0, len(queries), batch_size, desc="encode queries"):
        embeddings = wrapper.encode_queries(
            queries[start_idx : start_idx + batch_size],
            show_progress_bar=False,
            convert_to_tensor=True,
            batch_size=batch_size,
            max_length=query_max_length,
            max_new_tokens=1024,
            do_sample=False,
            instruction=instruction,
            reasons=reasons[start_idx : start_idx + batch_size] if isinstance(reasons, list) else None,
        )

        query_emb += embeddings.cpu().float().tolist()

    query_emb = torch.tensor(query_emb)
    print("query_emb shape:", query_emb.shape)
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    query_emb = F.normalize(query_emb, p=2, dim=1)
    scores = (query_emb @ doc_emb.T) * 100
    scores = scores.tolist()
    return get_scores(query_ids=query_ids, doc_ids=doc_ids, scores=scores, excluded_ids=excluded_ids)
