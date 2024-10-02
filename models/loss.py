import torch.nn.functional as F
import torch
from models.submodule import *
import torch.nn as nn

def isNaN(x):
    return x != x


def model_loss(disp_ests, disp_gt, mask):
    weights = [0.5, 0.7, 1.0, 1.3, 1.6]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], reduction='mean'))
    return sum(all_losses)

def l1_loss(rc_images, image):
    weights = [0.5, 0.7, 1.0, 1.3, 1.6]
    all_losses = []
    for rc_image, weight in zip(rc_images, weights):
        all_losses.append(weight * F.smooth_l1_loss(image, rc_image, reduction='mean'))
    return sum(all_losses)
