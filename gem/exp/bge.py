from typing import Any

import torch
from sentence_transformers import SentenceTransformer


# https://github.com/orionw/FollowIR/blob/master/models/bge/evaluate_bge.py#L18
class BGE:
    model_name = "BAAI/bge-large-en"

    def __init__(self, model_name: str | None = None, **kwargs: Any) -> None:
        self._model = SentenceTransformer(
            model_name or self.model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            **kwargs,
        )

    def encode_queries(
        self,
        texts: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool = True,
        instruction: str = "Represent this sentence for searching relevant passages: ",
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._model.encode(
            [instruction + query for query in texts],
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )

    def encode_corpus(
        self,
        texts: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )
