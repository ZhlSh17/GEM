import random
from typing import Any

from vllm import LLM, SamplingParams

from gem.syn.filter import Filter


_PROMPT = """\
# Given the following user query analysis
{reason}

# Your task
You are a data annotator. \
Based on the above analysis, draft a document that {relevance}.

**Important**:
- Preferred length: {length}
- Directly reply with the content **only**
- Do NOT disclose/explain why it is relevant or irrelevant

# Your document
"""


_PROMPT_WITH_EXAMPLE = """\
# Given the following user query analysis
{reason}

# Your task
You are a data annotator. \
Based on the above analysis, draft a document that {relevance}.

**Important**:
- Preferred length: {length}
- Directly reply with the content **only**
- Do NOT disclose/explain why it is relevant or irrelevant
- You can write in a similar style to the example below; however, make sure your document {relevance}

# Example document
{example}

# Your document
"""


class DocGenerator:

    def __init__(
        self,
        model: str | LLM,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        neg_only: bool = False,
        num_pos: int = 1,
        num_neg: int = 1,
        keep_highest: bool = False,
        prob_include_example: float = 0.5,
        prob_apply_intent_constraint: float = 0.5,
        seed=42,
        **kwargs: Any,
    ) -> None:
        self.llm = LLM(model, **kwargs) if isinstance(model, str) else model
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=1,
            seed=seed,
        )

        if not self.tokenizer.chat_template:
            raise NotImplementedError("expect chat models")

        self._var_lens = ["short", "medium", "long (less than 1000 words)"]
        self._var_lens_weights = [50, 45, 5]
        self._prob_include_example = prob_include_example
        self._random = random.Random(seed)
        self._pos = "is relevant"
        self._prob_apply_intent_constraint = prob_apply_intent_constraint
        self._intent_constraint = "is irrelevant because it fails to satisfy the search intent"
        self._criteria_constraint = "is irrelevant because it fails to meet some requirements"

        self.neg_only = neg_only
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.keep_highest = keep_highest

        self._filter = Filter(self.llm)

    @property
    def _neg(self) -> str:
        if self._random.random() < self._prob_apply_intent_constraint:
            return self._intent_constraint
        else:
            return self._criteria_constraint

    def _generate(
        self,
        batch: dict[str, list],
        num_samples: int,
        is_pos: bool = True,
        *,
        use_tqdm: bool = True,
    ) -> list[list[str]]:
        self.sampling_params.n = num_samples

        batch_msgs = []

        for idx in range(len(batch["query"])):
            query = batch["query"][idx]
            reason = batch["reason"][idx]
            example = None
            docs = batch["pos"][idx] if is_pos else batch["neg"][idx]
            if self._prob_include_example > 0 and docs:
                example = self._random.choice(docs)

            assert isinstance(query, str) and isinstance(reason, str)

            batch_msgs.append(
                self._get_msg(
                    query=query,
                    reason=reason,
                    relevance=self._pos if is_pos else self._neg,
                    example=example,
                )
            )

        docs = []
        queries = []
        reasons = []
        logprobs = []
        for idx, outputs in enumerate(self.llm.chat(batch_msgs, self.sampling_params, use_tqdm=use_tqdm)):
            if len(outputs.outputs) != num_samples:
                raise RuntimeError("unexpected misalignment")

            for output in outputs.outputs:
                queries.append(batch["query"][idx])
                reasons.append(batch["reason"][idx])
                docs.append(output.text)
                logprobs.append(output.cumulative_logprob)

        results = self._filter(queries=queries, reasons=reasons, docs=docs, use_tqdm=use_tqdm)
        masks = [is_relevant == is_pos for is_relevant in results]

        ret = []
        for idx in range(len(batch["query"])):
            _docs = []
            _logprobs = []
            for _idx in range(idx * num_samples, idx * num_samples + num_samples):
                if masks[_idx]:
                    _docs.append(docs[_idx])
                    _logprobs.append(logprobs[_idx])

            if _docs and self.keep_highest:
                idx_max_logprob = max(range(len(_logprobs)), key=lambda x: _logprobs[x])
                _docs = [_docs[idx_max_logprob]]

            ret.append(_docs)

        return ret

    def generate(self, batch: dict[str, list], *, use_tqdm: bool = True) -> list[dict]:
        batch_neg = self._generate(batch, num_samples=self.num_neg, is_pos=False, use_tqdm=use_tqdm)

        if not self.neg_only:
            batch_pos = self._generate(batch, num_samples=self.num_pos, is_pos=True, use_tqdm=use_tqdm)

        ret = []
        for idx in range(len(batch["query"])):
            item = {}
            for field in batch.keys():
                item[field] = batch[field][idx]

            if batch_neg[idx]:
                item["neg"] = batch_neg[idx]
                item["is_neg_syn"] = True
            else:
                item["is_neg_syn"] = False

            if not self.neg_only and batch_pos[idx]:
                item["pos"] = batch_pos[idx]
                item["is_pos_syn"] = True
            else:
                item["is_pos_syn"] = False

            ret.append(item)

        return ret

    def _get_msg(
        self,
        *,
        query: str,
        reason: str,
        relevance: str,
        example: str | None = None,
    ) -> list[dict]:
        length = self._random.sample(self._var_lens, k=1, counts=self._var_lens_weights)[0]

        kwargs = {"query": query, "reason": reason, "relevance": relevance, "length": length}
        if self._random.random() < self._prob_include_example and example:
            prompt = _PROMPT_WITH_EXAMPLE
            kwargs["example"] = example
        else:
            prompt = _PROMPT

        return [{"role": "user", "content": prompt.format(**kwargs)}]
