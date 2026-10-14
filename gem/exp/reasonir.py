from typing import Any

import torch
from transformers import AutoModel


# Adapted from: https://github.com/facebookresearch/ReasonIR#sentence-transformers
class ReasonIR:
    model_name = "reasonir/ReasonIR-8B"

    def __init__(
        self,
        model_name: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 2048,
    ) -> None:
        self._model = AutoModel.from_pretrained(
            model_name or self.model_name,
            trust_remote_code=True,
            dtype=dtype,
            device_map="auto",
        )
        self._model.eval()
        # See: https://github.com/facebookresearch/ReasonIR/blob/main/training/train.sh#L77
        self._max_length = max_length

    def encode_queries(
        self,
        texts: list[str],
        *,
        batch_size: int = 16,
        show_progress_bar: bool = True,
        max_length: int | None = None,
        instruction: str = "",
        **kwargs: Any,
    ) -> torch.Tensor:
        # See: https://huggingface.co/reasonir/ReasonIR-8B/blob/main/modeling_reasonir_8b.py#L1673
        return self._model.encode(
            texts,
            instruction=instruction,
            batch_size=batch_size,
            max_length=max_length or self._max_length,
            convert_to_tensor=True,
        )

    def encode_corpus(
        self,
        texts: list[str],
        *,
        batch_size: int = 16,
        show_progress_bar: bool = True,
        max_length: int | None = None,
        instruction: str = "",
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._model.encode(
            texts,
            instruction=instruction,
            batch_size=batch_size,
            max_length=max_length or self._max_length,
            convert_to_tensor=True,
        )
