from typing import Any

from vllm import LLM, SamplingParams


_FILTER_PROMPT = """\
# 1. User query analysis
{reason}

# 2. Example document
{doc}

# 3. Based on the analysis, is the example document relevant? Just answer Yes or No.
"""


class Filter:

    def __init__(self, model: str | LLM, **kwargs: Any) -> None:
        self.sampling_params = SamplingParams(temperature=0, max_tokens=16)
        self.llm = LLM(model, **kwargs) if isinstance(model, str) else model

    def __call__(
        self,
        reasons: list[str],
        docs: list[str],
        use_tqdm: bool = True,
        **kwargs: Any,
    ) -> list[bool]:
        if len(reasons) != len(docs):
            raise RuntimeError("length mismatch")

        messages = [
            [{"role": "user", "content": _FILTER_PROMPT.format(reason=reason, doc=doc)}]
            for reason, doc in zip(reasons, docs)
        ]

        outputs = self.llm.chat(messages, self.sampling_params, use_tqdm=use_tqdm)

        return ["yes" in output.outputs[0].text.lower() for output in outputs]
