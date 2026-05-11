import argparse
from abc import abstractmethod
from typing import Protocol, Any

import torch
import numpy as np

from gem.model import GEM


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


class BiEncoder(Protocol):

    @abstractmethod
    def encode_query(self, texts: list[str], /, batch_size: int, verbose: bool, **kwargs: Any) -> "torch.Tensor": ...

    @abstractmethod
    def encode_passage(self, texts: list[str], /, batch_size: int, verbose: bool, **kwargs: Any) -> "torch.Tensor": ...


def extract_corpus_sentences(corpus: list[dict[str, str]] | dict[str, list] | list[str], sep: str) -> list[str]:
    """Extracts sentences from the corpus"""
    if isinstance(corpus, dict):
        sentences = [
            (corpus["title"][i] + sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip()
            for i in range(len(corpus["text"]))
        ]

    elif isinstance(corpus, list):
        if isinstance(corpus[0], str):  # if corpus is a list of strings
            sentences = corpus
        else:
            sentences = [
                (doc["title"] + sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus
            ]
    return sentences


class BaseEncoder:

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


class GEM_Encoder(BaseEncoder):

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


def load_model(args: argparse.Namespace) -> BaseEncoder:
    model = None

    if args.model == "gem":
        if args.checkpoint is None:
            raise ValueError("require GEM checkpoint")

        model = GEM_Encoder(
            GEM.from_pretrained(args.checkpoint, device_map="auto", dtype=torch.bfloat16).eval(),
            no_generation=args.no_generation,
            generation_batch_size=args.generation_batch_size,
        )

    elif args.model == "repllama":
        from gem.exp.repllama import Repllama

        # sep=" " just to be consistent with official example: https://huggingface.co/castorini/repllama-v1-7b-lora-passage
        model = BaseEncoder(Repllama(), sep=" ")

    elif args.model == "promptriever":
        from gem.exp.promptriever import Promptriever

        model = BaseEncoder(Promptriever(), sep=" ")

    elif args.model == "reasonir":
        from gem.exp.reasonir import ReasonIR

        model = BaseEncoder(ReasonIR())

    else:
        raise NotImplementedError(f"unsupported model: {args.model}")

    return model
