from __future__ import annotations
 
import torch
import torch.nn as nn
 
 
class DiceBCELoss(nn.Module):
    """
    Combinaison de Binary Cross-Entropy et Dice Loss.
 
    La BCE assure une convergence stable, tandis que la Dice Loss
    gère le déséquilibre de classes (souvent plus de fond que de glacier).
 
    Parameters
    ----------
    bce_weight, dice_weight : float
        Poids relatifs des deux termes.
    smooth : float
        Terme de lissage pour éviter la division par zéro dans le Dice.
    """
 
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_w = bce_weight
        self.dice_w = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
 
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE (applique sigmoid en interne)
        bce = self.bce(logits, targets)
 
        # Dice
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
 
        return self.bce_w * bce + self.dice_w * dice.mean()
 
 
def iou_score(
    preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6
) -> float:
    """
    IoU (Intersection over Union) moyen sur un batch.
 
    Parameters
    ----------
    preds : Tensor (B, 1, H, W)
        Prédictions binaires (0 ou 1).
    targets : Tensor (B, 1, H, W)
        Masques de référence (0 ou 1).
 
    Returns
    -------
    float
        IoU moyen sur le batch.
    """
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - inter
    iou = (inter + smooth) / (union + smooth)
    return iou.mean().item()