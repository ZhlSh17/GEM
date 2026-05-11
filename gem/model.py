from typing import Any
from dataclasses import dataclass

import torch
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
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
from transformers.loss.loss_utils import ForCausalLMLoss as causal_lm_loss
from transformers.generation.utils import GenerateOutput
from loguru import logger
from tqdm import tqdm

# TODO: make this module standalone when push to huggingface model hub
from gem.train.loss import DistributedContrastiveLoss
from gem.preprocess import Preprocessor


@dataclass
class GEMOutput(ModelOutput):
    q_embs: torch.Tensor | None = None
    p_embs: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    gen_loss: torch.Tensor | None = None
    emb_loss: torch.Tensor | None = None


class GEM(PreTrainedModel, GenerationMixin):
    config_class = AutoConfig
    base_model_prefix = "_model"
    supports_gradient_checkpointing = True
    # may change depend on the backbone model
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        gen_loss_weight: float = 1.0,
        emb_loss_weight: float = 1.0,
        negatives_cross_device: bool = False,
        emb_loss_temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(config)

        self._model: PreTrainedModel = AutoModelForCausalLM.from_config(config, **kwargs)
        if isinstance(self._model, PreTrainedModel) and self._model.base_model_prefix == self.base_model_prefix:
            logger.warning(
                f"{GEM.base_model_prefix=} conflicts with the underlying base model {type(self._model)}. `from_pretrained` may not work as expected."
            )

        # TODO: train only config
        if gen_loss_weight < 0 or emb_loss_weight < 0:
            raise ValueError(f"{gen_loss_weight=}, {emb_loss_weight=}; expect non-negative values")

        if gen_loss_weight == 0 and emb_loss_weight == 0:
            raise ValueError("both gen_loss_weight and emb_loss_weight are 0; at least one loss should be optimized")

        if gen_loss_weight == 0 and emb_loss_weight != 1:
            emb_loss_weight = 1
            logger.info(f"{emb_loss_weight=} changed to 1 since {gen_loss_weight=}")
        elif emb_loss_weight == 0 and gen_loss_weight != 1:
            gen_loss_weight = 1
            logger.info(f"{gen_loss_weight=} changed to 1 since {emb_loss_weight=}")

        self._gen_loss_weight = gen_loss_weight
        self._emb_loss_weight = emb_loss_weight
        self._emb_loss_fn = DistributedContrastiveLoss(
            temperature=emb_loss_temperature,
            negatives_cross_device=negatives_cross_device,
        )

    @torch.no_grad()
    def generate(self, *args: Any, **kwargs: Any) -> GenerateOutput | torch.LongTensor:
        self._model.generation_config = self.generation_config
        return self._model.generate(*args, **kwargs)

    @torch.no_grad()
    def encode_passage(
        self,
        passage: str | list[str],
        *,
        instruction: str | None = None,
        batch_size: int = 16,
        verbose: bool = True,
        max_length: int | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> torch.Tensor:
        passages = [passage] if isinstance(passage, str) else passage
        tokenizer = tokenizer or self.tokenizer
        preproc = Preprocessor(tokenizer)

        if isinstance(max_length, int):
            passages = self._truncate(passages, max_length=max_length, tokenizer=tokenizer)

        texts = [preproc.preprocess_passage(x, instruction=instruction) for x in passages]
        return self.encode(texts, tokenizer=tokenizer, batch_size=batch_size, verbose=verbose)

    @torch.no_grad()
    def encode_query(
        self,
        query: str | list[str],
        reason: str | list[str] | None = None,
        *,
        instruction: str | None = None,
        batch_size: int = 16,
        verbose: bool = True,
        max_length: int | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        return_generation: bool = False,
        **generation_kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, list[str]]:
        """Encode queries.

        Args:
            query (str | list[str]): A single query or a list of queries.
            tokenizer (PreTrainedTokenizerBase | None, optional): Tokenizer. Defaults to None.
                If not provided, will load from the model's `config.name_or_path`.
            reason (str | list[str] | None, optional): Pre-computed reasonings. Defaults to None.
                If not provided, will generate on-the-fly.
            instruction (str | None, optional): Query instruction. Defaults to None.
                If not provided, a default instruction will be used.
            batch_size (int, optional): Batch size for encoding. Defaults to 32.
            verbose (bool, optional): Whether to show progress bar. Defaults to True.
            max_length (int, optional): Max number of query tokens for truncation. Defaults to None.
            return_generation (bool, optional): Whether to return reasonings. Defaults to False.
                Only effective when generation is performed.
            **generation_kwargs: Additonal keyword args for `model.generate`. E.g., `do_sample` and `max_new_tokens`.

        Raises:
            NotImplementedError: If pad_token_id set to eos_token_id.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list[str]]: A 2D embedding tensor with shape (batch_size, dim)
                if `return_generation=False`. When `return_generation=True`, returns an additional list of reasonings.
        """
        queries = [query] if isinstance(query, str) else query
        reasons = [reason] if isinstance(reason, str) else reason
        tokenizer = tokenizer or self.tokenizer
        preproc = Preprocessor(tokenizer)

        if isinstance(max_length, int):
            queries = self._truncate(queries, max_length=max_length, tokenizer=tokenizer)

        if reasons is not None:
            if len(queries) != len(reasons):
                raise RuntimeError(f"lengths of input `query` and `reason` lists mismatch")

            texts = [
                preproc.preprocess_query(query=query, reason=reason, instruction=instruction)
                for query, reason in zip(queries, reasons)
            ]
            return self.encode(texts, tokenizer=tokenizer, batch_size=batch_size, verbose=verbose)

        if (
            self.generation_config.pad_token_id is None
            or self.generation_config.pad_token_id == tokenizer.eos_token_id
        ):
            tokenizer.pad_token = tokenizer.bos_token
            self.generation_config.pad_token_id = tokenizer.pad_token_id
            # raise NotImplementedError(
            #     f"expect a valid `pad_token_id` that is different from `eos_token_id` in {self.generation_config=}"
            # )

        # batch generation and encoding
        all_embs = []
        all_reasons = []
        for start_idx in tqdm(
            range(0, len(queries), batch_size),
            desc="generate and encode",
            disable=not verbose,
        ):
            embs = self._generate_then_encode(
                queries=queries[start_idx : start_idx + batch_size],
                tokenizer=tokenizer,
                preproc=preproc,
                instruction=instruction,
                return_generation=return_generation,
                **generation_kwargs,
            )

            if return_generation:
                embs, reasons = embs
                all_reasons.extend(reasons)

            all_embs.append(embs)

        all_embs = torch.cat(all_embs, dim=0)
        return (all_embs, all_reasons) if return_generation else all_embs

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 16,
        verbose: bool = True,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> torch.Tensor:
        """Encode preprocessed texts with chat template applied and embedding token appended."""
        if isinstance(texts, str):
            texts = [texts]

        tokenizer = tokenizer or self.tokenizer

        all_embs = []
        for start_idx in tqdm(
            range(0, len(texts), batch_size),
            desc="encode",
            disable=not verbose,
        ):
            inputs = tokenizer(
                texts[start_idx : start_idx + batch_size],
                add_special_tokens=False,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            ).to(self._model.device)
            all_embs.append(self._get_embs(self._model(**inputs, output_hidden_states=True)))

        return torch.cat(all_embs, dim=0)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if not hasattr(self, "_tokenizer"):
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.config.name_or_path)
            except Exception as e:
                raise RuntimeError(
                    "failed to automatically load pretrained tokenizer; please manually set tokenizer"
                ) from e

        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    def forward(
        self,
        *,
        query: dict[str, torch.Tensor] | None = None,
        passage: dict[str, torch.Tensor] | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> GEMOutput:
        if self.training:
            return self._forward_train(query=query, passage=passage, labels=labels, **kwargs)
        else:
            raise NotImplementedError

    def _forward_train(
        self,
        *,
        query: dict[str, torch.Tensor],
        passage: dict[str, torch.Tensor],
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> GEMOutput:
        loss = None
        q_embs = None
        p_embs = None

        # total number of tokens
        num_items_in_batch = kwargs.pop("num_items_in_batch", None)
        # contrasitive loss is already averaged across devices when `negatives_cross_device=True`
        # the total effective batch size is therefore len(q_embs) * gradient_accumulation_steps
        gradient_accumulation_steps = kwargs.pop("gradient_accumulation_steps", None)

        gen_loss = 0.0
        # use_cache=False for FSDP
        q_outputs: CausalLMOutputWithPast = self._model(**query, output_hidden_states=True, use_cache=False)
        if labels is not None and self._gen_loss_weight != 0:
            gen_loss = causal_lm_loss(
                logits=q_outputs.logits,
                labels=labels,
                vocab_size=self.config.get_text_config().vocab_size,
                num_items_in_batch=num_items_in_batch,
            )
            if not gen_loss.isnan():
                # in case `labels` are all -100
                loss = self._gen_loss_weight * gen_loss

        emb_loss = 0.0
        if self._emb_loss_weight != 0:
            q_embs = self._get_embs(q_outputs)
            p_embs = self._get_embs(self._model(**passage, output_hidden_states=True, use_cache=False))
            emb_loss = self._emb_loss_fn(q_embs, p_embs)
            if isinstance(gradient_accumulation_steps, int) and gradient_accumulation_steps > 1:
                emb_loss = emb_loss / gradient_accumulation_steps

            if self._emb_loss_fn.is_ddp:
                # Trainer will mult loss by world_size when `compute_loss`
                emb_loss = emb_loss / self._emb_loss_fn.world_size

            if loss is None:
                loss = self._emb_loss_weight * emb_loss
            else:
                loss += self._emb_loss_weight * emb_loss

        if not isinstance(loss, torch.Tensor):
            raise RuntimeError("both gen_loss and emb_loss are None; at least one loss should be optimized")

        gen_loss = gen_loss.detach() if isinstance(gen_loss, torch.Tensor) else torch.tensor(gen_loss)
        emb_loss = emb_loss.detach() if isinstance(emb_loss, torch.Tensor) else torch.tensor(emb_loss)

        return GEMOutput(
            q_embs=q_embs,
            p_embs=p_embs,
            loss=loss,
            gen_loss=gen_loss,
            emb_loss=emb_loss,
        )

    def _get_embs(self, outputs: CausalLMOutputWithPast) -> torch.Tensor:
        last_hidden_state = outputs.hidden_states[-1]

        # we use left padding
        embs = last_hidden_state[:, -1, :]
        embs = nn.functional.normalize(embs, p=2, dim=-1)
        return embs

    def _generate_then_encode(
        self,
        queries: list[str],
        tokenizer: PreTrainedTokenizerBase,
        preproc: Preprocessor,
        instruction: str | None = None,
        return_generation: bool = False,
        **generation_kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, list[str]]:
        # Generate-then-encode by reusing KV cache
        # 1. create generation prompts
        texts = [
            preproc.preprocess_query(query=query, instruction=instruction, reason=None, append_embed_token=False)
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

        # 3. dynamically append eos_token (if not generated) and embedding token
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
                ),  # completion
                appended_inputs["attention_mask"],  # additional (for embedding)
            ],
            dim=-1,
        )
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.clamp_(min=0)
        position_ids = position_ids[:, -next_input_ids.size(1) :]

        # 5. another forward pass reusing KV cache
        embs = self._get_embs(
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

    def _truncate(
        self,
        texts: list[str],
        max_length: int,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> list[str]:
        tokenizer = tokenizer or self.tokenizer

        if isinstance(texts, str):
            texts = [texts]

        inputs = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )

        return tokenizer.batch_decode(inputs["input_ids"])
