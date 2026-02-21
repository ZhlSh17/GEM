import os
import json
from typing import Any
from pathlib import Path

import torch
from loguru import logger
from vllm import LLM, SamplingParams

from gem.model import GEM

# See paper: https://aclanthology.org/2023.acl-long.99.pdf
HYDE_PROMPT = """Please write a passage to answer the question.
Question: {query}
Passage:"""


# See paper: https://aclanthology.org/2023.emnlp-main.585.pdf
QUERY2DOC_PROMPT = """Write a passage that answers the given query:

Query: what state is this zip code 85282
Passage: Welcome to TEMPE, AZ 85282. 85282 is a rural zip code in Tempe, Arizona. The population
is primarily white, and mostly single. At $200,200 the average home value here is a bit higher than
average for the Phoenix-Mesa-Scottsdale metro area, so this probably isn’t the place to look for housing
bargains.5282 Zip code is located in the Mountain time zone at 33 degrees latitude (Fun Fact: this is the
same latitude as Damascus, Syria!) and -112 degrees longitude.

Query: why is gibbs model of reflection good
Passage: In this reflection, I am going to use Gibbs (1988) Reflective Cycle. This model is a recognised
framework for my reflection. Gibbs (1988) consists of six stages to complete one cycle which is able
to improve my nursing practice continuously and learning from the experience for better practice in the
future.n conclusion of my reflective assignment, I mention the model that I chose, Gibbs (1988) Reflective
Cycle as my framework of my reflective. I state the reasons why I am choosing the model as well as some
discussion on the important of doing reflection in nursing practice.

Query: what does a thousand pardons means
Passage: Oh, that’s all right, that’s all right, give us a rest; never mind about the direction, hang the
direction - I beg pardon, I beg a thousand pardons, I am not well to-day; pay no attention when I soliloquize,
it is an old habit, an old, bad habit, and hard to get rid of when one’s digestion is all disordered with eating
food that was raised forever and ever before he was born; good land! a man can’t keep his functions
regular on spring chickens thirteen hundred years old.

Query: what is a macro warning
Passage: Macro virus warning appears when no macros exist in the file in Word. When you open
a Microsoft Word 2002 document or template, you may receive the following macro virus warning,
even though the document or template does not contain macros: C:\<path>\<file name>contains macros.
Macros may contain viruses.

Query: {query}
Passage:"""


class HyDE:

    def __init__(
        self,
        model: str | LLM,
        temperature: float = 0,
        max_tokens: int = 1024,
        *,
        prompt: str = HYDE_PROMPT,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self.sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        self.llm = LLM(model, **kwargs) if isinstance(model, str) else model
        self.tokenizer = self.llm.get_tokenizer()
        self.prompt = prompt

    def generate(self, queries: list[str], *, use_tqdm: bool = True) -> list[str]:
        if self.tokenizer.chat_template:
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": self.prompt.format(query=query)}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for query in queries
            ]
        else:
            prompts = [self.prompt.format(query=query) for query in queries]

        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=self.sampling_params,
            use_tqdm=use_tqdm,
        )

        if len(outputs) != len(queries):
            raise RuntimeError("unexpected misalignment")

        return [output.outputs[0].text for output in outputs]

    def generate_and_save(self, queries: list[str], save_path: str | os.PathLike) -> None:
        if os.path.exists(save_path):
            raise FileExistsError(f"{save_path} exists")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        results = self.generate(queries, use_tqdm=True)

        with open(save_path, "wt") as f:
            json.dump({q: r for q, r in zip(queries, results)}, f, indent=2)

        logger.info(f"results saved to {save_path}")


class Query2Doc(HyDE):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs, prompt=kwargs.pop("prompt", QUERY2DOC_PROMPT))


class GEMReasoner:

    def __init__(
        self,
        model_name_or_path: str,
        temperature: float = 0,
        max_tokens: int = 1024,
        max_query_length: int = 1024,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = GEM.from_pretrained(model_name_or_path, device_map="auto", dtype=dtype).eval()

        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_query_length = max_query_length

    def generate(
        self,
        queries: list[str],
        batch_size: int = 4,
        *,
        use_tqdm: bool = True,
        **kwargs: Any,
    ) -> list[str]:
        if "max_length" not in kwargs:
            kwargs["max_length"] = self._max_query_length

        if "max_new_tokens" not in kwargs:
            kwargs["max_new_tokens"] = self._max_tokens

        if "do_sample" not in kwargs:
            kwargs["do_sample"] = self._temperature != 0

        if "temperature" not in kwargs:
            kwargs["temperature"] = self._temperature

        _, resps = self.model.encode_query(
            queries,
            batch_size=batch_size,
            return_generation=True,
            verbose=use_tqdm,
            **kwargs,
        )

        if len(resps) != len(queries):
            raise RuntimeError("unexpected misalignment")

        return resps

    def generate_and_save(self, queries: list[str], save_path: str | os.PathLike, **kwargs: Any) -> None:
        if os.path.exists(save_path):
            raise FileExistsError(f"{save_path} exists")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        results = self.generate(queries, use_tqdm=True, **kwargs)

        with open(save_path, "wt") as f:
            json.dump({q: r for q, r in zip(queries, results)}, f, indent=2)

        logger.info(f"results saved to {save_path}")
