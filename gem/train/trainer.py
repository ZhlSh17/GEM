from typing import Any
from collections import defaultdict
from dataclasses import dataclass

import torch
from transformers import Trainer
from transformers.utils.generic import ModelOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.loss.loss_utils import ForCausalLMLoss as causal_lm_loss
from loguru import logger

from gem.train.loss import DistributedContrastiveLoss
from gem.model import GEM


@dataclass
class GEMOutput(ModelOutput):
    q_embs: torch.Tensor | None = None
    p_embs: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    gen_loss: torch.Tensor | None = None
    emb_loss: torch.Tensor | None = None


class GEMTrainer(Trainer):

    def __init__(
        self,
        *args: Any,
        gen_loss_weight: float = 0.1,
        emb_loss_weight: float = 1.0,
        negatives_cross_device: bool = False,
        emb_loss_temperature: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.args.remove_unused_columns = False

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

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any | dict | torch.Tensor]:
        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
                if model.training:
                    kwargs["gradient_accumulation_steps"] = self.current_gradient_accumulation_steps

            inputs = {**inputs, **kwargs}

        # outputs = model(**inputs)
        outputs = self._compute_loss_gem(model, **inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        loss_mult = None
        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss_mult = self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu
            # 1. generation loss is already averaged across devices through `num_items_in_batch`
            # 2. for embedding loss, we have scaled it in `model.forward` to counter this
            loss *= loss_mult

        self._log_losses(outputs, loss_mult, model.training)

        return (loss, outputs) if return_outputs else loss

    def _log_losses(self, outputs: Any, loss_mult: int | None = None, training: bool = True) -> None:
        if not isinstance(outputs, dict):
            return

        state = self.state

        if not hasattr(state, "_loss_accum_sums"):
            state._loss_accum_sums = defaultdict(float)
            state._loss_accum_count = 0

        for key in ("emb_loss", "gen_loss"):
            loss = outputs.get(key, None)
            if loss is None:
                continue

            if not isinstance(loss, torch.Tensor):
                raise ValueError(f"{key} should be a torch.Tensor, but got {type(loss)}")

            loss = loss.detach().mean()
            state._loss_accum_sums[key] += loss

        state._loss_accum_count += 1

        if training and not self.accelerator.sync_gradients:
            return

        data = {}

        if self.args.world_size >= 1:
            for key, total in state._loss_accum_sums.items():
                # average across processes
                value = self._nested_gather(total).mean().item()
                if loss_mult is not None:
                    value *= loss_mult
                data[key] = value

        if data:
            # only the main process will log
            self.log(data)

        # clean up for next training step
        state._loss_accum_sums.clear()
        state._loss_accum_count = 0

    def _compute_loss_gem(
        self,
        model: GEM,
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

        q_outputs: CausalLMOutputWithPast = model(**query, output_hidden_states=True, use_cache=False)
        if labels is not None and self._gen_loss_weight != 0:
            gen_loss = causal_lm_loss(
                logits=q_outputs.logits,
                labels=labels,
                vocab_size=model.config.get_text_config().vocab_size,
                num_items_in_batch=num_items_in_batch,
            )
            if not gen_loss.isnan():
                # in case `labels` are all -100
                loss = self._gen_loss_weight * gen_loss

        emb_loss = 0.0
        if self._emb_loss_weight != 0:
            q_embs = model.last_token_pool(q_outputs)
            p_embs = model.last_token_pool(model(**passage, output_hidden_states=True, use_cache=False))
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
