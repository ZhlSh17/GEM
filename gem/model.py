from typing import Any

import torch
import numpy as np
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationMixin,
    PreTrainedModel,
    PretrainedConfig,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.utils import logging
from tqdm import tqdm

from gem.preprocess import Preprocessor

logger = logging.get_logger(__name__)


class GEM(PreTrainedModel, GenerationMixin):
    config_class = AutoConfig
    base_model_prefix = "_model"
    supports_gradient_checkpointing = True
    # can be adjusted based on the backbone
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def __init__(self, config: PretrainedConfig, **kwargs: Any) -> None:
        super().__init__(config)

        self._model: PreTrainedModel = AutoModelForCausalLM.from_config(config, **kwargs)
        if isinstance(self._model, PreTrainedModel) and self._model.base_model_prefix == self.base_model_prefix:
            logger.warning(
                f"{GEM.base_model_prefix=} conflicts with the underlying base model {type(self._model)}. `from_pretrained` may not work as expected."
            )

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if not hasattr(self, "_tokenizer"):
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.config.name_or_path)
            except Exception as e:
                raise RuntimeError("failed to automatically load pretrained tokenizer; set it manually") from e

        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    @staticmethod
    def last_token_pool(outputs: CausalLMOutputWithPast) -> torch.Tensor:
        last_hidden_state = outputs.hidden_states[-1]

        # we use left padding
        embs = last_hidden_state[:, -1, :]
        embs = nn.functional.normalize(embs, p=2, dim=-1)
        return embs

    def forward(self, *args: Any, **kwargs: Any):
        return self._model.forward(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args: Any, **kwargs: Any) -> GenerateOutput | torch.LongTensor:
        self._model.generation_config = self.generation_config
        return self._model.generate(*args, **kwargs)

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 16,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor | np.ndarray:
        """Encode preprocessed texts.

        This is a low-level function for both query-side and doc-side encoding,
        assuming that the chat template has been applied and the embedding token has been appended.

        See Also:
            :meth:`GEM.generate_then_encode`

        Returns:
            torch.Tensor | np.ndarray: 2D tensor of shape (batch_size, dim).

        """
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
                add_special_tokens=False,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            ).to(self._model.device)
            all_embs.append(self.last_token_pool(self._model(**inputs, output_hidden_states=True)))

        embs = torch.cat(all_embs, dim=0)
        return embs if convert_to_tensor else embs.cpu().float().numpy()

    def encode_corpus(
        self,
        corpus: list[str],
        *,
        batch_size: int = 16,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = True,
        instruction: str | None = None,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | np.ndarray:
        """Encode corpus.

        Returns:
            torch.Tensor | np.ndarray: 2D tensor of shape (batch_size, dim).

        """
        if isinstance(corpus, str):
            corpus = [corpus]

        preproc = Preprocessor(self.tokenizer)

        if isinstance(max_length, int):
            corpus = self._truncate(corpus, max_length=max_length)

        corpus = [preproc.preprocess_passage(x, instruction=instruction) for x in corpus]
        return self.encode(
            corpus,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=convert_to_tensor,
        )

    def encode_queries(
        self,
        queries: list[str],
        *,
        batch_size: int = 16,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = True,
        reasons: list[str] | None = None,
        instruction: str | None = None,
        prompt: str | None = None,
        max_length: int | None = None,
        return_generation: bool = False,
        **generation_kwargs: Any,
    ) -> torch.Tensor | np.ndarray | tuple[torch.Tensor | np.ndarray, list[str]]:
        """Encode queries.

        Args:
            queries (list[str]): A list of queries.
            reasons (list[str] | None, optional): Pre-computed reasoning.
                If not provided, will generate on-the-fly via `generate_then_encode`. Defaults to None.
            instruction (str | None, optional): Instruction.
                If not provided, a default instruction will be used. Effective only when `prompt` is not provided. Defaults to None.
            prompt (str | None, optional): Prompt template.
                This allows fully customized input format. Expect a prompt template with a `{text}` placeholder, e.g. "some meta-instruction ... {text}".
                If not provided, a default prompt template will be used. Defaults to None.
            batch_size (int, optional): Batch size for encoding. Defaults to 16.
            show_progress_bar (bool, optional): Whether to show progress bar. Defaults to True.
            max_length (int, optional): Max number of query tokens for truncation. Defaults to None.
            return_generation (bool, optional): Whether to return generated reasoning.
                Effective only when generation is performed. Defaults to False.
            **generation_kwargs: Additonal keyword args for `model.generate`. E.g., `do_sample`, `temperature` and `max_new_tokens`.

        Returns:
            torch.Tensor | np.ndarray | tuple[torch.Tensor | np.ndarray, list[str]]: A 2D embedding tensor of shape (batch_size, dim), with an optional list of reasoning.
                - torch.Tensor: `return_generation=False`, `convert_to_tensor=True`
                - np.ndarray: `return_generation=False`, `convert_to_tensor=False`
                - tuple[torch.Tensor, list[str]]: `return_generation=True`, `convert_to_tensor=True`
                - tuple[np.ndarray, list[str]]: `return_generation=True`, `convert_to_tensor=False`

        """
        queries = [queries] if isinstance(queries, str) else queries
        reasons = [reasons] if isinstance(reasons, str) else reasons

        if isinstance(max_length, int):
            queries = self._truncate(queries, max_length=max_length)

        if reasons is not None:
            if len(queries) != len(reasons):
                raise RuntimeError(f"{len(queries)=} does not align with {len(reasons)=}")

            preproc = Preprocessor(self.tokenizer)
            texts = [
                preproc.preprocess_query(query=query, reason=reason, instruction=instruction, prompt=prompt)
                for query, reason in zip(queries, reasons)
            ]

            return self.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_tensor=convert_to_tensor,
            )

        return self.generate_then_encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=convert_to_tensor,
            instruction=instruction,
            prompt=prompt,
            return_generation=return_generation,
            **generation_kwargs,
        )

    @torch.no_grad()
    def generate_then_encode(
        self,
        queries: list[str],
        *,
        batch_size: int = 16,
        show_progress_bar: bool = True,
        convert_to_tensor: bool = True,
        instruction: str | None = None,
        prompt: str | None = None,
        return_generation: bool = False,
        **generation_kwargs: Any,
    ) -> torch.Tensor | np.ndarray | tuple[torch.Tensor | np.ndarray, list[str]]:
        """Generate-then-encode.

        See Also:
            :meth:`GEM.encode_queries`

        """
        if (
            self.generation_config.pad_token_id is None
            or self.generation_config.pad_token_id == self.tokenizer.eos_token_id
        ):
            logger.warning(
                "`generate_then_encode` requires a valid pad token that differs from eos token. "
                "Note that the bos token is now used as the default for generation padding. "
                "You may manually update the pad token and generation config if needed."
            )
            self.tokenizer.pad_token = self.tokenizer.bos_token
            self.generation_config.pad_token_id = self.tokenizer.pad_token_id

        if isinstance(queries, str):
            queries = [queries]

        all_embs = []
        all_reasons = []
        for start_idx in tqdm(
            range(0, len(queries), batch_size),
            desc="generate-then-encode",
            disable=not show_progress_bar,
        ):
            embs = self._generate_then_encode(
                queries=queries[start_idx : start_idx + batch_size],
                instruction=instruction,
                prompt=prompt,
                return_generation=return_generation,
                **generation_kwargs,
            )

            if return_generation:
                embs, reasons = embs
                all_reasons.extend(reasons)

            all_embs.append(embs)

        all_embs = torch.cat(all_embs, dim=0)
        all_embs = all_embs if convert_to_tensor else all_embs.cpu().float().numpy()
        return (all_embs, all_reasons) if return_generation else all_embs

    def _generate_then_encode(
        self,
        queries: list[str],
        instruction: str | None = None,
        prompt: str | None = None,
        return_generation: bool = False,
        **generation_kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, list[str]]:
        tokenizer = self.tokenizer
        preproc = Preprocessor(tokenizer)

        # Generate-then-encode by reusing KV cache
        # 1. prompts
        texts = [
            preproc.preprocess_query(
                query=query,
                instruction=instruction,
                reason=None,
                append_embed_token=False,
                prompt=prompt,
            )
            for query in queries
        ]
        inputs = tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        ).to(self._model.device)

        # 2. generation
        if (
            isinstance(generation_kwargs.get("num_return_sequences"), int)
            and generation_kwargs.get("num_return_sequences") != 1
        ):
            logger.warning("`generate_then_encode` does not support beam search; set `num_return_sequences` to 1")
            generation_kwargs["num_return_sequences"] = 1

        outputs = self.generate(
            **inputs,
            use_cache=True,
            return_dict_in_generate=True,
            **generation_kwargs,
        )
        # remove prompt tokens
        generated_ids = outputs.sequences[:, inputs["input_ids"].size(1) :]
        # shorter generated sequences can be padded
        gen_attention_mask = generated_ids != self.generation_config.pad_token_id

        # 3. dynamically append eos_token (if not generated, due to `max_new_tokens`) and embedding token
        # format conforms to `tokenizer.chat_template`
        appended_texts = []
        for idx in range(generated_ids.size(0)):
            gen_sample_ids = generated_ids[idx][gen_attention_mask[idx]]  # remove pad tokens
            gen_sample = tokenizer.decode(gen_sample_ids, skip_special_tokens=False)
            gen_sample_skip_st = tokenizer.decode(gen_sample_ids, skip_special_tokens=True)
            appended_texts.append(
                preproc.preprocess_query(
                    query=queries[idx],
                    instruction=instruction,
                    reason=gen_sample_skip_st,  # avoid duplicate eos_token
                    prompt=prompt,
                )[len(texts[idx]) + len(gen_sample) :]
            )

        appended_inputs = tokenizer(
            appended_texts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            padding_side="left",
        ).to(generated_ids.device)

        # 4. (re-)construct valid input_ids, attention_mask, position_ids
        next_input_ids = torch.cat([outputs.sequences[:, -1:], appended_inputs["input_ids"]], dim=-1)
        attention_mask: torch.Tensor = inputs["attention_mask"]
        attention_mask = torch.cat(
            [
                attention_mask,  # prompt
                gen_attention_mask.to(
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),  # response
                appended_inputs["attention_mask"],  # additional (for embedding)
            ],
            dim=-1,
        )
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.clamp_(min=0)
        position_ids = position_ids[:, -next_input_ids.size(1) :]

        # 5. another forward pass reusing KV cache
        embs = self.last_token_pool(
            self._model(
                input_ids=next_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=outputs.past_key_values,
                output_hidden_states=True,
            )
        )

        if return_generation:
            return embs, tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

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
