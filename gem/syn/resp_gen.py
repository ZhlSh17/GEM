import random
from typing import Any

from vllm import LLM, SamplingParams

from gem.preprocess import QUERY_INSTRUCTION, Preprocessor
from gem.syn.filter import Filter


class RespGenerator:

    def __init__(
        self,
        model: str | LLM,
        n: int = 1,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        *,
        instruction: str = QUERY_INSTRUCTION,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self.sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )

        self.llm = LLM(model, **kwargs) if isinstance(model, str) else model
        self.tokenizer = self.llm.get_tokenizer()

        self.instruction = instruction

        self._preproc = Preprocessor(self.tokenizer)
        self._random = random.Random(seed)

        self._filter = Filter(self.llm)

    def generate(self, batch: dict[str, list], *, use_tqdm: bool = True) -> list[dict]:
        batch_queries: list[str] = batch["query"]
        batch_pos: list[list[str]] = batch["pos"]
        batch_neg: list[list[str]] = batch["neg"]
        batch_source: list[str] = batch["_source"]
        assert len(batch_queries) == len(batch_pos) == len(batch_neg) == len(batch_source)

        batch_completions = self._reason(batch_queries, use_tqdm=use_tqdm)
        batch_pos = [self._sample_doc(docs) for docs in batch_pos]

        batch_completions_pos = self._apply_filter(
            batch_queries=batch_queries,
            batch_completions=batch_completions,
            batch_docs=batch_pos,
            is_pos=True,
            use_tqdm=use_tqdm,
        )

        data = []
        for idx in range(len(batch_completions)):
            item = {
                "_source": batch_source[idx],
                "query": batch_queries[idx],
                "instruction": self.instruction,
                "pos": batch_pos[idx],
                "neg": batch_neg[idx],
                "reason": None,
            }

            raw_completions = batch_completions[idx]
            if not raw_completions:
                data.append(item)
                continue

            completions_pos = batch_completions_pos[idx]

            if completions_pos:
                item["reason"] = self._random.choice(completions_pos)

            data.append(item)

        return data

    def _apply_filter(
        self,
        batch_queries: list[str],
        batch_completions: list[list[str]],
        batch_docs: list[list[str]],
        is_pos: bool,
        use_tqdm: bool = True,
    ) -> list[list[str]]:
        queries = []
        reasons = []
        docs = []
        idx2len = {}

        for idx, (query, doc, completions) in enumerate(zip(batch_queries, batch_docs, batch_completions)):
            if not (isinstance(doc, list) and len(doc) == 1 and isinstance(doc[0], str)):
                raise RuntimeError(f"unexpected schema: {doc=}")

            doc = doc[0]

            if not isinstance(completions, list):
                raise RuntimeError(f"unexpected schema: {completions=}")

            idx2len[idx] = len(completions)
            for reason in completions:
                reasons.append(reason)
                docs.append(doc)
                queries.append(query)

        if not reasons:
            return [[] for _ in range(len(idx2len))]

        masks = []
        results = self._filter(queries=queries, reasons=reasons, docs=docs, use_tqdm=use_tqdm)
        start = 0
        for idx in range(len(idx2len)):
            end = start + idx2len[idx]
            if start == end:
                masks.append(None)
            else:
                masks.append(results[start:end])
            start = end

        ret_completions = []
        for mask, completions in zip(masks, batch_completions):
            if mask is None:
                ret_completions.append([])
                continue

            if not isinstance(completions, list) or len(mask) != len(completions):
                raise RuntimeError(f"unexpected misalignment")

            ret_completions.append([compl for is_relevant, compl in zip(mask, completions) if is_relevant == is_pos])

        if len(ret_completions) != len(batch_completions):
            raise RuntimeError(f"unexpected misalignment")

        return ret_completions

    def _reason(self, queries: list[str], use_tqdm: bool = True) -> list[list[str]]:
        prompts = [
            self._preproc.preprocess_query(
                query,
                instruction=self.instruction,
                reason=None,
                append_embed_token=False,
            )
            for query in queries
        ]
        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=self.sampling_params,
            use_tqdm=use_tqdm,
        )

        if len(queries) != len(outputs):
            raise RuntimeError("unexpected misalignment")

        batch_completions = []
        for output in outputs:
            completions = []
            for item in output.outputs:
                if item.finish_reason == "stop":
                    completions.append(item.text)

            batch_completions.append(completions)

        return batch_completions

    def _sample_doc(self, docs: list[str]) -> list[str]:
        if len(docs) == 1:
            return docs

        if len(docs) > 1:
            return self._random.sample(docs, k=1)

        raise RuntimeError("require at least 1 documents")
