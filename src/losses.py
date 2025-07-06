import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)
        device = inputs.device

        targets_one_hot = F.one_hot(targets, n_classes).float()
        targets_smooth = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / n_classes

        p = F.softmax(inputs, dim=-1)
        ce_loss = -targets_smooth * torch.log(p + 1e-8)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t).pow(self.gamma)

        focal_loss = focal_weight.unsqueeze(1) * ce_loss
        return focal_loss.sum(dim=1).mean()