"""Training loss: weighted cross-entropy + Dice + focal (unchanged from the study).

The cross-entropy term covers the whole crop; Dice and focal count only the
cells that were NOT already burning on day t (the "spread region"), which is
what the model is asked to predict.
"""
import torch
import torch.nn.functional as F


def combined_loss(output, target, spread_mask, pos_weight=4.0, bce_weight=0.4,
                  dice_weight=0.6, focal_weight=0.1, gamma=4.0):
    """output: logits (B,1,H,W); target, spread_mask: (B,1,H,W) of 0/1.
    Returns (total, bce, dice, focal)."""
    bce = F.binary_cross_entropy_with_logits(
        output, target, pos_weight=torch.tensor(pos_weight, device=output.device))
    pred = torch.sigmoid(output).clamp(1e-6, 1 - 1e-6)
    inter = (pred * target * spread_mask).sum(dim=(2, 3))
    union = ((pred + target) * spread_mask).sum(dim=(2, 3))
    dice = (1 - (2 * inter + 1e-4) / (union + 1e-4)).mean()
    pt = torch.where(target > 0.5, pred, 1 - pred)
    focal = ((-((1 - pt) ** gamma) * torch.log(pt)) * spread_mask).mean()
    total = bce_weight * bce + dice_weight * dice + focal_weight * focal
    if torch.isnan(total) or torch.isinf(total):
        total = 0.5 * (bce + dice)
    return total, bce, dice, focal
