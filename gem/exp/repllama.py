from typing import Any

import torch
from torch import nn
from peft import PeftModel, PeftConfig
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from tqdm import tqdm


# Adapted from: https://huggingface.co/castorini/repllama-v1-7b-lora-passage
class Repllama:
    model_name: str = "castorini/repllama-v1-7b-lora-passage"

    def __init__(
        self,
        model_name: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        model_name = model_name or self.model_name
        config = PeftConfig.from_pretrained(model_name)
        self.peft_config = config

        base_model = AutoModel.from_pretrained(config.base_model_name_or_path, device_map="auto", dtype=dtype)
        model = PeftModel.from_pretrained(base_model, model_name, device_map="auto", dtype=dtype)
        model = model.merge_and_unload()
        model.eval()
        self._model: PreTrainedModel = model

        # right padding by default
        tokenizer = AutoTokenizer.from_pretrained(self.peft_config.base_model_name_or_path)
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.pad_token = tokenizer.unk_token
        self.tokenizer: PreTrainedTokenizerBase = tokenizer

    @torch.no_grad()
    def encode_queries(
        self,
        texts: list[str],
        *,
        prompt: str = "query: {}</s>",
        batch_size: int = 16,
        show_progress_bar: bool = True,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if isinstance(max_length, int):
            texts = self._truncate(texts, max_length=max_length)

        texts = [prompt.format(x) for x in texts]
        return self.encode(texts, batch_size=batch_size, show_progress_bar=show_progress_bar)

    @torch.no_grad()
    def encode_corpus(
        self,
        texts: list[str],
        *,
        prompt: str = "passage: {}</s>",
        batch_size: int = 16,
        show_progress_bar: bool = True,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if isinstance(max_length, int):
            texts = self._truncate(texts, max_length=max_length)

        texts = [prompt.format(x) for x in texts]
        return self.encode(texts, batch_size=batch_size, show_progress_bar=show_progress_bar)

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 16,
        show_progress_bar: bool = True,
    ) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        all_embs = []
        for start_idx in tqdm(
            range(0, len(texts), batch_size),
            desc="encode",
            disable=not show_progress_bar,
        ):
            inputs = self.tokenizer(
                texts[start_idx : start_idx + batch_size],
                padding=True,
                padding_side="left",
                return_tensors="pt",
            ).to(self._model.device)
            all_embs.append(self._get_embs(self._model(**inputs, output_hidden_states=True)))

        return torch.cat(all_embs, dim=0)

    def _get_embs(self, outputs: CausalLMOutputWithPast) -> torch.Tensor:
        last_hidden_state = outputs.hidden_states[-1]

        # we use left padding
        embs = last_hidden_state[:, -1, :]
        embs = nn.functional.normalize(embs, p=2, dim=-1)
        return embs

    def _truncate(self, texts: list[str], max_length: int) -> list[str]:
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )

        return self.tokenizer.batch_decode(inputs["input_ids"])
