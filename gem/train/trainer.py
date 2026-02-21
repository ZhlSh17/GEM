from typing import Any
from collections import defaultdict

import torch
from transformers import Trainer


class GEMTrainer(Trainer):

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

        outputs = model(**inputs)
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
