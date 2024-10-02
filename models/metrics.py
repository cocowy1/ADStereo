import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_occlusion_mask(disparity):
    # disparity in left view
    _, _, _, W = disparity.shape
    index = torch.arange(0, W, device=disparity.device, requires_grad=False).view(1, 1, 1, W)
    matching_right = index - disparity      # W_r = W_l - Disp
    # invisible region
    visible = (matching_right > 0.).float() # B1HW
    # occlusion region
    count = 0.
    for i in range(1, W):
        shift_map = F.pad(matching_right[:, :, :, i:], (0, i, 0, 0), mode='constant', value=-1)     # shift in left
        count = count + (torch.abs(matching_right - shift_map) < 0.5).float()                       # 0.5 means round
    occlud = (count > 0.).float()
    # TODO: 增加形态学处理去除孔洞
    # 最终得到的不包含invisible和occlusion的mask
    valid_mask = visible * (1 - occlud)
    return valid_mask.bool().float().detach()


def get_edge_mask(disparity, thresh=10., dilation=10):
    # disparity in left view, similarity to SMD-Nets
    def gradient_x(img):
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gx

    def gradient_y(img):
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gy

    # 一阶梯度
    gx_disp = torch.abs(gradient_x(disparity))
    gy_disp = torch.abs(gradient_y(disparity))

    # 得到边缘
    edges = (gx_disp > thresh).float() + (gy_disp > thresh).float()
    edges = (edges > 0.).float()

    # 对边缘进行膨胀, 以考虑边缘附近的像素估计精度
    if dilation > 0:
        edge_list = []
        kernel = np.ones((dilation, dilation), np.uint8)
        for i in range(disparity.shape[0]):
            edge_slice = cv2.dilate(edges[i,0,...].data.cpu().numpy(), kernel, iterations=1)
            edge_list.append(edge_slice)
        edges = np.stack(edge_list, axis=0)         # HW -> BHW
        edges = torch.from_numpy(edges)             # To Tensor
        edges = torch.unsqueeze(edges, dim=1)       # B1HW
        edges = edges.to(disparity.device)          # change device

    return edges




def IoUpoint(target, output, **args):
    """
    input args:
    target: G.T. unocclusion mask, 0-1 float
    output: P.T. unocclusion mask, 0-1 float

    output args:
    ExtraPoint:   实际被遮挡, 但是计算出了深度值的点, 用于分析边缘生长出多少. 采用 IoU 计算方法;
    InvalidPoint: 未被遮挡, 但是没算出深度值的点;
    """
    # print((((1 - target) * output) > 1e-3).sum())
    # print(((1 - target) > 1e-3).sum())
    ExtraPoint = (((1 - target) * output) > 1e-3).float().sum() / ((1 - target) > 1e-3).float().sum()       # (GT_occlu ∩ output_valid) / GT_occlu
    InvalidPoint = ((target * (1 - output)) > 1e-3).float().sum() / (target > 1e-3).float().sum()             # 1 - (GT_valid ∩ output_valid) / GT_valid
    return ExtraPoint, InvalidPoint


def epe_metric(target, output, mask):
    """
        target: G.T. disparity or depth, tensor, B1HW
        output: P.T. disparity or depth, tensor, B1HW
        valid:  A mask, which region use for calculate error, B1HW, bool
    """
    target, output = target[mask], output[mask]                 # 考虑有效区域
    err = torch.abs(target - output)                            # L1 误差
    avg = torch.mean(err)                                       # 误差均值
    return avg.data.cpu()


def d1_metric(target, output, mask, threshold=3):
    target, output = target[mask], output[mask]                 # 考虑有效区域
    err = torch.abs(target - output)                            # L1 误差
    err_mask = (err > threshold) & (err / target > 0.05)        # 超过3像素, 或相对估计误差大于5%, 视为 D1 误差
    err_mean = torch.mean(err_mask.float())                     # (0,1), 百分比
    return err_mean.data.cpu()                      


class METRICS(nn.Module):
    def __init__(self):
        super(METRICS, self).__init__()
        self.max_disp = 0
        self.min_disp = 192

    def forward(self, target, output):

        # 考虑所有候选视差范围内的结果
        gt_concern = (target > 0) & (target < 191)
        gt_concern.detach_()

        # 若预测视差非稠密, 则统计有效部分
        pt_concern = output > 0
        pt_concern.detach_()
        # if only_visible:
        #     visible = get_occlusion_mask(target)    # 排除不可见区域的结果
        # else:
        visible = 1.

        # 得到最终要估计的区域
        mask = gt_concern.float() * pt_concern.float() * visible
        # if torch.sum(mask) == 0:
        #     # Following GwcNet, we remove all the images with less than 10% valid pixels (0≤d<Dmax) in the test set
        #     metrics = {"EPE": 0., "Bad0.5": 0., "Bad1.0": 0., "Bad3.0": 0.}
        # else:
        #     mask = mask.bool()
        #     epe  = epe_metric(target, output, mask)
        #     bad0 = d1_metric(target, output, mask, threshold=0.5)
        #     bad1 = d1_metric(target, output, mask, threshold=1.0)
        #     bad2 = d1_metric(target, output, mask, threshold=3.0)
        #     metrics = {"EPE": epe.item(), "Bad0.5": bad0.item(), "Bad1.0": bad1.item(), "Bad3.0": bad2.item()}

        edge_mask = get_edge_mask(disparity=target, thresh=1., dilation=5)
        eval_mask = mask.float() * edge_mask
        if torch.sum(eval_mask) == 0:
            edge_error = 0.
        else:
            # TODO: following SMD-Nets to use soft edge error
            eval_mask = eval_mask.bool()
            edge_error = epe_metric(target, output, eval_mask).item()

        return edge_error
