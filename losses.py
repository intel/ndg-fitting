import random

import torch
from ext import pytorch_ssim


class DssimL1Loss(torch.nn.Module):

    def __init__(self, b=0.2, permute=True):
        super(DssimL1Loss, self).__init__()
        # DSSIM and L1 blending factor
        self.b = b
        # Whether to permute, i.e. whether input -> (B, H, W, C)
        self.permute = permute

        self.loss_map = torch.zeros([0])

    def forward(self, pred, gt):
        dssim = (1.0 - pytorch_ssim.ssim(pred, gt, size_average=False, permute=self.permute))
        l1 = torch.abs(pred-gt)
        self.loss_map = (1.0 - self.b) * l1 + self.b * dssim

        return self.loss_map.mean()


class RelL2Loss(torch.nn.Module):

    def __init__(self):
        super(RelL2Loss, self).__init__()
        self.loss_map = torch.zeros([0])

    def forward(self, pred, gt):
        self.loss_map = (pred-gt)**2/((pred.detach()**2)+0.01)

        return self.loss_map.mean()


class PSNRLoss(torch.nn.Module):

    def __init__(self):
        super(PSNRLoss, self).__init__()
        self.loss_map = torch.zeros([0])

    def forward(self, pred, gt):
        self.loss_map = ((pred - gt) ** 2).view(pred.shape[0], -1).mean(1, keepdim=True)

        return 20 * torch.log10(1.0 / torch.sqrt(self.loss_map))


class L1Loss(torch.nn.Module):

    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss_map = torch.zeros([0])

    def forward(self, pred, gt):
        self.loss_map = torch.abs(pred-gt)
        return self.loss_map.mean()


class L2Loss(torch.nn.Module):

    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss_map = torch.zeros([0])

    def forward(self, pred, gt):
        self.loss_map = (pred - gt)**2
        return self.loss_map.mean()


class SMAPELoss(torch.nn.Module):

    def __init__(self):
        super(SMAPELoss, self).__init__()
        self.loss_map = torch.zeros([0])

    def forward(self, pred, gt):
        self.loss_map = torch.abs(pred-gt)/(torch.abs(pred)+torch.abs(gt)+0.01)

        return self.loss_map.mean()


class AllMetrics(torch.nn.Module):

    def __init__(self):
        super(AllMetrics, self).__init__()

        self.samples = 0

        self.metrics = {}

        self.metrics['l1'] = 0
        self.metrics['l2'] = 0
        self.metrics['lpips'] = 0
        self.metrics['dssim'] = 0
        self.metrics['psnr'] = 0
        self.metrics['mape'] = 0
        self.metrics['smape'] = 0
        self.metrics['mrse'] = 0

        self.loss_map = torch.zeros([0])

    def reset(self):
        self.samples = 0

        self.metrics['l1'] = 0
        self.metrics['l2'] = 0
        self.metrics['dssim'] = 0
        self.metrics['psnr'] = 0
        self.metrics['mape'] = 0
        self.metrics['smape'] = 0
        self.metrics['mrse'] = 0

    def forward(self, pred, gt):
        self.samples += 1

        diff = gt - pred
        eps = 1e-2

        self.metrics['l1'] += (torch.abs(diff)).mean()
        self.metrics['l2'] += (diff*diff).mean()
        self.metrics['mrse'] += (diff*diff/(gt*gt+eps)).mean()
        self.metrics['mape'] += (torch.abs(diff)/(gt+eps)).mean()
        self.metrics['smape'] += (2 * torch.abs(diff)/(gt+pred+eps)).mean()
        self.metrics['dssim'] += (1.0 - (pytorch_ssim.ssim(pred, gt, size_average=False, permute=True))).mean()
        self.metrics['psnr'] += (20.0 * torch.log10(1.0/(torch.sqrt(((gt.clip(0, 1) - pred.clip(0, 1))**2).mean()))))

        return self

    def __add__(self, other):
        self.samples += 1

        self.metrics['l1'] = self.metrics['l1'] + other.metrics['l1']
        self.metrics['l2'] = self.metrics['l2'] + other.metrics['l2']
        self.metrics['dssim'] = self.metrics['dssim'] + other.metrics['dssim']
        self.metrics['mape'] = self.metrics['mape'] + other.metrics['mape']
        self.metrics['smape'] = self.metrics['smape'] + other.metrics['smape']
        self.metrics['mrse'] = self.metrics['mrse'] + other.metrics['mrse']
        self.metrics['lpips'] = self.metrics['lpips'] + other.metrics['lpips']
        self.metrics['psnr'] = self.metrics['psnr'] + other.metrics['psnr']
        return self

    def __truediv__(self, n):
        self.metrics['l1'] = self.metrics['l1'] / n
        self.metrics['l2'] = self.metrics['l2'] / n
        self.metrics['dssim'] = self.metrics['dssim'] / n
        self.metrics['mape'] = self.metrics['mape'] / n
        self.metrics['smape'] = self.metrics['smape'] / n
        self.metrics['mrse'] = self.metrics['mrse'] / n
        self.metrics['lpips'] = self.metrics['lpips'] / n
        self.metrics['psnr'] = self.metrics['psnr'] / n
        return self



