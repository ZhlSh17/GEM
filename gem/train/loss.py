from typing import Optional

import torch
import torch.distributed as dist


# Adapted from: http://github.com/ContextualAI/gritlm/blob/main/gritlm/training/model.py#L25
class DistributedContrastiveLoss:

    def __init__(self, temperature: float, negatives_cross_device: bool):
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
        self.temperature = temperature
        self.negatives_cross_device = negatives_cross_device

        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        if self.negatives_cross_device:
            if not self.is_ddp:
                raise ValueError("Cannot do negatives_cross_device without distributed training")

    def __call__(self, q_reps: torch.Tensor, p_reps: torch.Tensor) -> torch.Tensor:
        if self.negatives_cross_device:
            # This gathers both negatives and positives.
            # It could likely be optimized by only gathering negatives.
            q_reps = self._dist_gather_tensor(q_reps)
            p_reps = self._dist_gather_tensor(p_reps)
        scores = self.compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)

        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target *= p_reps.size(0) // q_reps.size(0)
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        # All tensors have the same shape, as pooling already applied to them
        dist.all_gather(all_tensors, t)

        all_tensors[self.rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))
