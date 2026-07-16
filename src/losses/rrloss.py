"""BCE3 + recursive-refinement loss, ported from external/cmrrwnet/train/losses.py.

Channel order (matches src/biomarkers/labels.py prediction format and
GaveAVDataset's label tensor): index0=artery, index1=vessel(all), index2=vein.
"""
import torch
import torch.nn as nn


class BCE3Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_vessels: torch.Tensor, vessels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = torch.round(mask[:, 0, :, :])

        pred_a, pred_vt, pred_v = pred_vessels[:, 0], pred_vessels[:, 1], pred_vessels[:, 2]
        gt_a, gt_vt, gt_v = vessels[:, 0], vessels[:, 1], vessels[:, 2]

        loss = self.loss(pred_a[mask > 0.5], gt_a[mask > 0.5])
        loss = loss + self.loss(pred_v[mask > 0.5], gt_v[mask > 0.5])
        loss = loss + self.loss(pred_vt[mask > 0.5], gt_vt[mask > 0.5])
        return loss

    def process_predicted(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(prediction.clone())


class RRLoss(nn.Module):
    """Weighted sum of per-iteration BCE3 losses; later refinement iterations
    weighted more heavily (weight index i for predictions[2:])."""

    def __init__(self, base_criterion: nn.Module):
        super().__init__()
        self.base_criterion = base_criterion

    def forward(self, predictions: list[torch.Tensor], gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss_1 = self.base_criterion(predictions[0], gt, mask)
        if len(predictions) == 1:
            return loss_1

        loss_2 = self.base_criterion(predictions[1], gt, mask)
        if len(predictions) == 2:
            return loss_1 + loss_2
        for i, prediction in enumerate(predictions[2:], 2):
            loss_2 = loss_2 + i * self.base_criterion(prediction, gt, mask)

        k = len(predictions[1:])
        z = 0.5 * k * (k + 1)
        loss_2 = loss_2 / z

        return loss_1 + loss_2

    def process_predicted(self, predictions: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self.base_criterion.process_predicted(p) for p in predictions]
