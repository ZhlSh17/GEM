import os

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, PeftConfig


@torch.no_grad()
def retrieval_repllama(
    queries,
    query_ids,
    documents,
    doc_ids,
    task,
    model_id,
    instructions,
    cache_dir,
    excluded_ids,
    long_context,
    **kwargs,
):
    """Run BRIGHT retrieval using repllama.

    Adapted from:
        https://github.com/xlang-ai/BRIGHT/blob/main/retrievers.py#L131

    See Also:
        https://huggingface.co/castorini/repllama-v1-7b-lora-passage
    """
    from third_party.ReasonIR.evaluation.bright.retrievers import last_token_pool, get_scores

    config = PeftConfig.from_pretrained("castorini/repllama-v1-7b-lora-passage")
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path, device_map="auto")
    model = PeftModel.from_pretrained(base_model, "castorini/repllama-v1-7b-lora-passage")
    model = model.merge_and_unload()
    model.eval()

    # right padding by default
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.pad_token = tokenizer.unk_token

    # format
    queries = [f"query: {x}</s>" for x in queries]
    documents = [f"passage: {x}</s>" for x in documents]

    # more config
    batch_size = kwargs.get("encode_batch_size", 1)
    max_length = kwargs.get("doc_max_length", 8192)

    # The following code is copied from https://github.com/xlang-ai/BRIGHT/blob/main/retrievers.py#L154
    doc_emb = None
    cache_path = os.path.join(cache_dir, "doc_emb", model_id, task, f"long_{long_context}_{batch_size}.npy")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.isfile(cache_path):
        # already exists so we can just load it
        doc_emb = np.load(cache_path, allow_pickle=True)

    for start_idx in trange(0, len(documents), batch_size):
        assert doc_emb is None or doc_emb.shape[0] % batch_size == 0, f"{doc_emb % batch_size} reminder in doc_emb"
        if doc_emb is not None and doc_emb.shape[0] // batch_size > start_idx:
            continue

        batch_dict = tokenizer(
            documents[start_idx : start_idx + batch_size],
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"]).cpu()
        # doc_emb[start_idx] = embeddings
        doc_emb = embeddings if doc_emb is None else np.concatenate((doc_emb, np.array(embeddings)), axis=0)

        # save the embeddings every 1000 iters, you can adjust this as needed
        if (start_idx + 1) % 1000 == 0:
            np.save(cache_path, doc_emb)

    np.save(cache_path, doc_emb)

    doc_emb = torch.tensor(doc_emb)
    print("doc_emb shape:", doc_emb.shape)
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    query_emb = []
    for start_idx in trange(0, len(queries), batch_size):
        batch_dict = tokenizer(
            queries[start_idx : start_idx + batch_size],
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"]).cpu().tolist()
        query_emb += embeddings
    query_emb = torch.tensor(query_emb)
    print("query_emb shape:", query_emb.shape)
    query_emb = F.normalize(query_emb, p=2, dim=1)
    scores = (query_emb @ doc_emb.T) * 100
    scores = scores.tolist()
    return get_scores(query_ids=query_ids, doc_ids=doc_ids, scores=scores, excluded_ids=excluded_ids)
