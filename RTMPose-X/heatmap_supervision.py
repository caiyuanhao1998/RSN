"""Optional spatial supervision of the existing stride-16 RSN feature maps.

This is a training loss, not a new inference head. Missing MPII face labels and
out-of-crop joints receive zero weight, including their background pixels.
"""
import math

import torch
from torch.nn import functional as F


def gaussian_targets(keypoints, visible, heatmap_size, input_size=(288, 384), sigma=1.):
    """Map input-pixel (x, y) coordinates to a (height, width) heatmap grid."""
    height, width = heatmap_size
    input_width, input_height = input_size
    if min(height, width, input_width, input_height) <= 0 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError('Heatmap/input sizes and sigma must be positive and finite')
    if keypoints.ndim != 3 or keypoints.shape[-1] != 2 or visible.shape != keypoints.shape[:2]:
        raise ValueError('Expected keypoints [batch, joints, 2] and visibility [batch, joints]')
    points = keypoints.detach().float()
    in_crop = (torch.isfinite(points).all(-1) & (points[..., 0] >= 0) &
               (points[..., 0] < input_width) & (points[..., 1] >= 0) &
               (points[..., 1] < input_height))
    weights = torch.where(in_crop, visible.detach().float().clamp_min(0), 0.)
    points = torch.nan_to_num(points)
    x = torch.arange(width, device=points.device, dtype=torch.float32)
    y = torch.arange(height, device=points.device, dtype=torch.float32)
    cx = points[..., 0, None, None]*(width/input_width)
    cy = points[..., 1, None, None]*(height/input_height)
    distance = (x[None, None, None, :]-cx).square() + (y[None, None, :, None]-cy).square()
    target = torch.exp(-distance/(2*sigma*sigma))
    return target, weights


def heatmap_loss(prediction, keypoints, visible, batch_reference=512,
                 input_size=(288, 384), sigma=1., ohkm_topk=0):
    """Weighted MSE with the same fixed local-batch scale as official SimCC.

    Using batch_reference times a local sample mean keeps the loss and accumulated
    gradient invariant when the physical minibatch changes. Mean mode uses the fixed
    joint denominator. Optional original-RSN-style OHKM averages the hardest top-k
    weighted joints per person; masked joints contribute zero even when selected.
"""
    if prediction.ndim != 4 or prediction.shape[:2] != visible.shape:
        raise ValueError('Expected heatmaps [batch, joints, height, width]')
    if batch_reference <= 0 or prediction.shape[0] == 0:
        raise ValueError('Batch sizes must be positive')
    if not 0 <= ohkm_topk <= prediction.shape[1]:
        raise ValueError('OHKM top-k must be between zero and the joint count')
    target, weights = gaussian_targets(keypoints, visible, prediction.shape[-2:], input_size, sigma)
    per_joint = (prediction.float()-target).square().mean((-2, -1))
    if ohkm_topk:
        return (per_joint*weights).topk(ohkm_topk, dim=-1).values.mean()*batch_reference
    return (per_joint*weights).mean()*batch_reference


def heatmap_distribution_loss(prediction, keypoints, visible, logit_scale, sigma=1.5, *,
                              foreground_mask=None, foreground_visible=None, foreground_ratio=0.):
    """Supervise a normalized spatial distribution on annotated, in-crop joints."""
    target, weights = gaussian_targets(keypoints, visible, prediction.shape[-2:], sigma=sigma)
    target = target.flatten(2)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    logits = prediction.float().flatten(2) * logit_scale
    log_probability = F.log_softmax(logits, dim=-1)
    loss = (F.kl_div(log_probability, target, reduction='none').sum(dim=-1) * weights).mean()
    if foreground_ratio:
        mass = (log_probability.exp() * foreground_mask.flatten(1)[:, None]).sum(-1)
        foreground = -(mass.clamp_min(1e-8).log() * weights * foreground_visible).mean()
        loss = loss + foreground_ratio * foreground
    return loss
