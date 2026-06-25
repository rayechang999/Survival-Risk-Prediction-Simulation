from __future__ import annotations

from typing import Optional

import torch
from torch.nn import MarginRankingLoss


def sampled_margin_ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.5,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Pairwise sampled margin ranking loss.
      - shuffle the target order with a random permutation
      - assign +1 when target_i > target_perm_i, else -1
      - compare prediction_i vs prediction_perm_i with MarginRankingLoss

    `predictions` and `targets` are flattened internally. The function is fully
    differentiable with respect to `predictions`.
    """
    pred = predictions.reshape(-1)
    tgt = targets.reshape(-1).to(pred.dtype)

    if pred.numel() != tgt.numel():
        raise ValueError("predictions and targets must have the same number of elements")
    if pred.numel() < 2:
        return pred.new_zeros(())

    if generator is None:
        index = torch.randperm(tgt.numel(), device=tgt.device)
    else:
        index = torch.randperm(tgt.numel(), generator=generator, device=tgt.device)

    diff = tgt - tgt[index]
    values = torch.where(diff > 0.02, 1.0, torch.where(diff < -0.02, -1.0, 0.0))    
    criterion = MarginRankingLoss(margin=float(margin))
    return criterion(pred, pred[index], values)


def deterministic_margin_ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.5,
    seed: int = 0,
) -> torch.Tensor:
    """Deterministic variant for evaluation/reporting."""
    generator = torch.Generator(device=predictions.device if predictions.device.type != "cpu" else "cpu")
    generator.manual_seed(int(seed))
    return sampled_margin_ranking_loss(predictions, targets, margin=margin, generator=generator)
