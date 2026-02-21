from abc import abstractmethod
from typing import Protocol, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class BiEncoder(Protocol):

    @abstractmethod
    def encode_query(self, texts: list[str], /, batch_size: int, verbose: bool, **kwargs: Any) -> "torch.Tensor": ...

    @abstractmethod
    def encode_passage(self, texts: list[str], /, batch_size: int, verbose: bool, **kwargs: Any) -> "torch.Tensor": ...
