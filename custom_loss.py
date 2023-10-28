# -*- encoding: utf-8 -*-
import torch.nn as nn
import torch


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='none')
        self.smoothL1 = nn.SmoothL1Loss(reduction='mean', beta=1.0)
        self.MSELoss_func_no_reduce = nn.MSELoss(reduction='none')
        self.MSELoss_func = nn.MSELoss()

    def forward(self, pred, gt):
        pred_A = pred[0]
        pred_B = pred[1]
        gt_dose = gt[0]

        MSE_loss_no_reduction = 0.5 * self.MSELoss_func_no_reduce(pred_A, gt_dose) + self.MSELoss_func_no_reduce(pred_B, gt_dose)
        weighted_loss = torch.mean(MSE_loss_no_reduction)


        return weighted_loss