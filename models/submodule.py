from __future__ import print_function

import time

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np
import math
# from models.kernels.lib_tree_filter.modules.tree_filter import MinimumSpanningTree
# from models.kernels.lib_tree_filter.modules.tree_filter import TreeFilter2D
import torchvision.models as models
from models.lib.nn import SynchronizedBatchNorm2d, SynchronizedBatchNorm3d

BatchNorm2d = SynchronizedBatchNorm2d
BatchNorm3d = SynchronizedBatchNorm3d
BN_MOMENTUM = 0.1


def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


def norm_correlation(fea1, fea2):
    cost = torch.mean(((fea1/(torch.norm(fea1, 2, 1, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 1, True)+1e-05))), dim=1, keepdim=True)
    return cost

def build_norm_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = norm_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = norm_correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume

def SpatialTransformer_grid(x, y, disp_range_samples):

    bs, channels, height, width = y.size()
    ndisp = disp_range_samples.size()[1]

    mh, mw = torch.meshgrid([torch.arange(0, height, dtype=x.dtype, device=x.device),
                                 torch.arange(0, width, dtype=x.dtype, device=x.device)])  # (H *W)

    mh = mh.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)
    mw = mw.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)  # (B, D, H, W)

    cur_disp_coords_y = mh
    cur_disp_coords_x = mw - disp_range_samples

    coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
    coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
    grid = torch.stack([coords_x, coords_y], dim=4) #(B, D, H, W, 2)

    y_warped = F.grid_sample(y, grid.view(bs, ndisp * height, width, 2), mode='bilinear',
                               padding_mode='zeros', align_corners=True).view(bs, channels, ndisp, height, width)  #(B, C, D, H, W)

    x_warped = x.unsqueeze(2).repeat(1, 1, ndisp, 1, 1) #(B, C, D, H, W)

    return y_warped, x_warped




class Conv2x_v(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super(Conv2x_v, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x
    
    
    
class BasicConv_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super(BasicConv_IN, self).__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x


class Conv2x_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, IN=True, relu=True, keep_dispc=False):
        super(Conv2x_IN, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv_IN(out_channels*2, out_channels*mul, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv_IN(out_channels, out_channels, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x
    
    
    
class Conv2dUnit(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.
    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu
    Notes:
        Default momentum for batch normalization is set to be 0.01,
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1,
                 relu=True, lrelu=False, bn=True, bn_momentum=0.1, gn=False, gn_group=32, **kwargs):
        super(Conv2dUnit, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              dilation=dilation, bias=(not bn and not gn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.gn = nn.GroupNorm(gn_group, out_channels) if gn else None
        self.relu = relu
        self.lrelu = lrelu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        elif self.gn is not None:
            x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        if self.lrelu:
            x = F.leaky_relu(x, negative_slope=0.1, inplace=True)
        return x


class Deconv2dUnit(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.
       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu
       Notes:
           Default momentum for batch normalization is set to be 0.01,
       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, gn=False, gn_group=32, **kwargs):
        super(Deconv2dUnit, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2, 3], "the stride({}) should be in [1,2,3]".format(stride)
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn and not gn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.gn = nn.GroupNorm(gn_group, out_channels) if gn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        # if self.stride == 3:
        # h, w = list(x.size())[2:]
        # y = y[:, :, :3 * h, :3 * w].contiguous()
        if self.bn is not None:
            x = self.bn(x)
        elif self.gn is not None:
            x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x



class Adaptive_filter(nn.Module):

    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect', group=2):
        super(Adaptive_filter, self).__init__()
        self.pad = get_pad_layer(pad_type)(kernel_size//2)
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(in_channels, group*kernel_size*kernel_size, kernel_size=kernel_size, stride=1, bias=False)
        self.bn = BatchNorm2d(group*kernel_size*kernel_size, momentum=BN_MOMENTUM)
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        sigma = self.conv(self.pad(x))
        sigma = self.bn(sigma)
        sigma = self.softmax(sigma)

        n,c,h,w = sigma.shape

        sigma = sigma.reshape(n,1,c,h*w)

        n,c,h,w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size,h*w))

        n,c1,p,q = x.shape
        x = x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4)

        n,c2,p,q = sigma.shape
        sigma = sigma.permute(2,0,1,3).reshape((p//(self.kernel_size*self.kernel_size), self.kernel_size*self.kernel_size,n,c2,q)).permute(2,0,3,1,4)

        x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w)
        return x[:,:,torch.arange(h)%self.stride==0,:][:,:,:,torch.arange(w)%self.stride==0]


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNormRelu, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          BatchNorm2d(int(n_filters), momentum=BN_MOMENTUM),
                                          nn.LeakyReLU(0.1, inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.LeakyReLU(0.1, inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class pyramidPooling(nn.Module):

    def __init__(self, in_channels, pool_sizes, model_name='pspnet', fusion_mode='cat', with_bn=True):
        super(pyramidPooling, self).__init__()

        bias = not with_bn

        self.paths = []
        if pool_sizes is None:
            for i in range(4):
                self.paths.append(conv2DBatchNormRelu(in_channels, in_channels, 1, 1, 0, bias=bias, with_bn=with_bn))
        else:
            for i in range(len(pool_sizes)):
                self.paths.append(
                    conv2DBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=bias,
                                        with_bn=with_bn))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes
        self.model_name = model_name
        self.fusion_mode = fusion_mode

    # @profile
    def forward(self, x):
        h, w = x.shape[2:]

        k_sizes = []
        strides = []
        if self.pool_sizes is None:
            for pool_size in np.linspace(2, min(h, w), 4, dtype=int):
                k_sizes.append((int(h / pool_size), int(w / pool_size)))
                strides.append((int(h / pool_size), int(w / pool_size)))
            k_sizes = k_sizes[::-1]
            strides = strides[::-1]
        else:
            k_sizes = [(self.pool_sizes[0], self.pool_sizes[0]), (self.pool_sizes[1], self.pool_sizes[1]),
                       (self.pool_sizes[2], self.pool_sizes[2]), (self.pool_sizes[3], self.pool_sizes[3])]
            strides = k_sizes

        if self.fusion_mode == 'cat':  # pspnet: concat (including x)
            output_slices = [x]

            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                # out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != 'icnet':
                    out = module(out)
                out = F.upsample(out, size=(h, w), mode='bilinear')
                output_slices.append(out)

            return torch.cat(output_slices, dim=1)
        else:  # icnet: element-wise sum (including x)
            pp_sum = x

            for i, module in enumerate(self.path_module_list):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                out = module(out)
                out = F.upsample(out, size=(h, w), mode='bilinear')
                pp_sum = pp_sum + 0.25 * out
            #pp_sum = nn.LeakyReLU(pp_sum / 2., inplace=True)
            pp_sum = FMish(pp_sum / 2.)
            return pp_sum

def FMish(x):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''

    return x * torch.tanh(F.softplus(x))

def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         BatchNorm2d(out_channels, momentum=BN_MOMENTUM))


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         BatchNorm3d(out_channels, momentum=BN_MOMENTUM))


def topkpool(cost, disp_samples, k):
    _, ind = cost.sort(1, True)
    pool_ind = ind[:, :k, ...]
    cv = torch.gather(cost, 1, pool_ind)
    disp = torch.gather(disp_samples, 1, pool_ind)

    return cv, disp

def regression(cost, disp_samples, k):
    b, _, h, w = cost.shape
    cost, disp_samples = topkpool(cost, disp_samples, k)
    pred_possibility = F.softmax(cost, 1)
    disparity_topk = torch.sum(pred_possibility * disp_samples, dim=1, keepdim=True)
    return disparity_topk

def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)

def regression_topk(cost, disparity_samples, k):
    _, ind = cost.sort(1, True)
    pool_ind = ind[:, :k]
    cost = torch.gather(cost, 1, pool_ind)
    prob = F.softmax(cost, 1)
    disparity_samples = torch.gather(disparity_samples, 1, pool_ind)
    pred = torch.sum(disparity_samples * prob, dim=1, keepdim=True)
    return pred

def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume

class MobileV2_Residual(nn.Module):
    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1):
        super(MobileV2_Residual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        pad = dilation

        if expanse_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim, momentum=BN_MOMENTUM),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup, momentum=BN_MOMENTUM),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim, momentum=BN_MOMENTUM),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim, momentum=BN_MOMENTUM),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup, momentum=BN_MOMENTUM),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class hourglass2D(nn.Module):
    def __init__(self, in_channels):
        super(hourglass2D, self).__init__()

        self.expanse_ratio = 2

        self.conv1 = MobileV2_Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)

        self.conv2 = MobileV2_Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

        self.conv3 = MobileV2_Residual(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)

        self.conv4 = MobileV2_Residual(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            BatchNorm2d(in_channels * 2, momentum=BN_MOMENTUM))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            BatchNorm2d(in_channels, momentum=BN_MOMENTUM))

        self.redir1 = MobileV2_Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
        self.redir2 = MobileV2_Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
            self.norm2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
            if not stride == 1:
                self.norm3 = BatchNorm2d(planes, momentum=BN_MOMENTUM)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
#        print(in_channels, out_channels, deconv, is_3d, bn, relu, kwargs)
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = BatchNorm3d(out_channels, momentum=BN_MOMENTUM)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = BatchNorm2d(out_channels, momentum=BN_MOMENTUM)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            BatchNorm2d(channels, momentum=BN_MOMENTUM),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            BatchNorm2d(channels, momentum=BN_MOMENTUM),
        )

    def forward(self, x):
        out = self.body(x)
        return self.lrelu(out + x)

# Encoder Block
class EncoderB(nn.Module):
    def __init__(self, n_blocks, channels_in, channels_out, downsample=False):
        super(EncoderB, self).__init__()
        body = []
        if downsample:
            body.append(nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, 2, 1, bias=False),
                BatchNorm2d(channels_out, momentum=BN_MOMENTUM),
                nn.LeakyReLU(0.1, inplace=True),
            ))
        if not downsample:
            body.append(nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, 1, 1, bias=False),
                BatchNorm2d(channels_out, momentum=BN_MOMENTUM),
                nn.LeakyReLU(0.1, inplace=True),
            ))
        for i in range(n_blocks):
            body.append(
                ResB(channels_out)
            )
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


# Decoder Block
class DecoderB(nn.Module):
    def __init__(self, n_blocks, channels_in, channels_out):
        super(DecoderB, self).__init__()
        body = []
        body.append(nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
                BatchNorm2d(channels_out, momentum=BN_MOMENTUM),
                nn.LeakyReLU(0.1, inplace=True),
            ))
        for i in range(n_blocks):
            body.append(
                ResB(channels_out)
            )
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)

class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

def warp_right_to_left(x, disp, warp_grid=None):
    B, C, H, W = x.size()
    # mesh grid
    if warp_grid is not None:
        xx0, yy = warp_grid
        xx = xx0 + disp
        xx = 2.0 * xx / max(W - 1, 1) - 1.0
    else:
        # xx = torch.arange(0, W, device=disp.device).float()
        # yy = torch.arange(0, H, device=disp.device).float()
        xx = torch.arange(0, W, device=disp.device, dtype=x.dtype)
        yy = torch.arange(0, H, device=disp.device, dtype=x.dtype)
        # if x.is_cuda:
        #    xx = xx.cuda()
        #    yy = yy.cuda()
        xx = xx.view(1, -1).repeat(H, 1)
        yy = yy.view(-1, 1).repeat(1, W)

        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)

        # apply disparity to x-axis
        xx = xx + disp
        xx = 2.0 * xx / max(W - 1, 1) - 1.0
        yy = 2.0 * yy / max(H - 1, 1) - 1.0

    grid = torch.cat((xx, yy), 1)

    vgrid = grid
    # vgrid[:, 0, :, :] = vgrid[:, 0, :, :] + disp[:, 0, :, :]
    # vgrid[:, 0, :, :].add_(disp[:, 0, :, :])
    # vgrid.add_(disp)

    # scale grid to [-1,1]
    # vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:] / max(W-1,1)-1.0
    # vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:] / max(H-1,1)-1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    # mask = torch.autograd.Variable(torch.ones_like(x))
    # mask = nn.functional.grid_sample(mask, vgrid)

    # mask[mask<0.9999] = 0
    # mask[mask>0] = 1

    # return output*mask
    return output  # *mask

def channel_length(x):
    return torch.sqrt(torch.sum(torch.pow(x, 2), dim=1, keepdim=True) + 1e-3)


class GlobalEncoder(nn.Module):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, in_plane):
        super(GlobalEncoder, self).__init__()
        self.inplanes = in_plane
        self.layer1 = self._make_layer(models.resnet.BasicBlock, in_plane, 2, stride=1)
        self.layer2 = self._make_layer(models.resnet.BasicBlock, in_plane, 2, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        norm_layer = BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

class DispConv(nn.Module):
    def __init__(self, channel):
        super(DispConv, self).__init__()
        body = []
        body.append(nn.Sequential(
        nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
        BatchNorm2d(channel, momentum=BN_MOMENTUM),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(channel, 1, 3, 1, 1, bias=False),
        ))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


class Gradient_feat(nn.Module):
    def __init__(self, output_dim=32, norm_fn='batch'):
        super(Gradient_feat, self).__init__()
        self.norm_fn = norm_fn

        self.grad = Get_gradient()

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=16)

        elif self.norm_fn == 'batch':
            self.norm1 = BatchNorm2d(16, momentum=BN_MOMENTUM)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(16)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        ## 4x use ##
        self.conv_start = nn.Sequential(nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3),
                                        self.norm1, nn.ReLU(inplace=True))
        ## 2x use ##
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        # self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 16
        self.layer1 = self._make_layer(16, stride=1)
        self.layer2 = self._make_layer(32, stride=2)

        # output convolution
        self.conv_g0 = BasicConv(32, 32, kernel_size=3, padding=1)
        self.output = nn.Conv2d(32, output_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.grad(x)
        x = self.conv_start(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.conv_g0(x)
        feat_grad = self.output(x)
        return feat_grad

class Get_gradient_disp(nn.Module):
    def __init__(self):
        super(Get_gradient_disp, self).__init__()
        kernel_v = np.array([[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]], dtype=np.float32)
        kernel_h = np.array([[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]], dtype=np.float32)

        kernel_h = torch.from_numpy(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.from_numpy(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)

    def forward(self, x):
        x_v = F.conv2d(x, self.weight_v.cuda(), padding=1)
        x_h = F.conv2d(x, self.weight_h.cuda(), padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)
        return x

class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = np.array([[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]], dtype=np.float32)
        kernel_h = np.array([[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]], dtype=np.float32)

        kernel_h = torch.from_numpy(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.from_numpy(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]

        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v.cuda(), padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h.cuda(), padding=1)
        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v.cuda(), padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h.cuda(), padding=1)
        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v.cuda(), padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h.cuda(), padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)

        return x


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]

        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding = 1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding = 1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding = 1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding = 1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding = 1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding = 1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)

        return x


def convbn_2d_lrelu(in_planes, out_planes, kernel_size, stride, pad, dilation=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=(kernel_size, kernel_size),
                  stride=(stride, stride), padding=(pad, pad), dilation=(dilation, dilation), bias=bias),
        BatchNorm2d(out_planes, momentum=BN_MOMENTUM),
        nn.LeakyReLU(0.1, inplace=True))


def convbn_2d_Tanh(in_planes, out_planes, kernel_size, stride, pad, dilation=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=(kernel_size, kernel_size),
                  stride=(stride, stride), padding=(pad, pad), dilation=(dilation, dilation), bias=bias),
        BatchNorm2d(out_planes, momentum=BN_MOMENTUM),
        nn.Tanh())

class Bilateral_filter(SubModule):
    def __init__(self, channel):
        super(Bilateral_filter, self).__init__()
        self.Slice = Slice()
        self.Guide = GuideNN()
        self.Coeff = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            BatchNorm2d(channel, momentum=BN_MOMENTUM),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, 1, 3, 1, 1, bias=False)
        )

    def forward(self, guide, feat, disp):
        guide = self.Guide(guide)
        coeff = self.Coeff(feat)
        coffes = self.Slice(coeff, guide)
        disp_refine = torch.sum(disp*coffes, dim=1, keepdim=True)

        return disp_refine


class Slice(SubModule):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)]) # [0,511] HxW
        hg = hg.cuda()
        wg = wg.cuda()
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) * 2 - 1 # norm to [-1,1] NxHxWx1

        bilateral_grid = bilateral_grid.unsqueeze(1)
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()  # [B,C,H,W]-> [B,H,W,C]
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1)  # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, align_corners=False)
        return coeff.squeeze(2)  # [B,1,H,W]

# class GuideNN(SubModule):
#     def __init__(self, params=None):
#         super(GuideNN, self).__init__()
#         self.params = params
#         self.conv1 = convbn_2d_lrelu(32, 64, 3, 2, 1)
#         self.conv2 = convbn_2d_lrelu(64, 64, 3, 1, 1)
#         self.conv3 = convbn_2d_Tanh(64, 1, 1, 1, 0)
#
#     def forward(self, x):
#         return self.conv3(self.conv2(self.conv1(x)))

# class GuideNN(SubModule):
#     def __init__(self, in_channels, out_channels):
#         super(GuideNN, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#
#         self.conv1 = convbn_2d_lrelu(self.in_channels, self.in_channels*2, 3, 2, 1)
#         self.conv2 = convbn_2d_lrelu(self.in_channels*2, self.in_channels*2, 3, 1, 1)
#         self.conv3 = nn.Conv2d(self.in_channels*2, self.out_channels, 1, 1, 0, bias=False)
#
#     def forward(self, x):
#         return self.conv3(self.conv2(self.conv1(x)))

class GuideNN(SubModule):
    def __init__(self, in_channels, out_channels):
        super(GuideNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = convbn_2d_lrelu(self.in_channels, self.in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        return self.conv2(self.conv1(x))




class RefinementNet_bilateral(SubModule):
    def __init__(self, input_channel, channels):
        super(RefinementNet_bilateral, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.downsample = nn.AvgPool2d(2)

        self.E0 = EncoderB(1, input_channel, channels[0], downsample=False)  # scale: 1/4
        self.E1 = EncoderB(1, channels[0], channels[1], downsample=True)  # scale: 1/8
        self.E2 = EncoderB(1, channels[1], channels[2], downsample=True)  # scale: 1/16

        # self.mst1 = MinimumSpanningTree(TreeFilter2D.norm2_distance, 64,  3)
        # self.tree_filter1 = TreeFilter2D(groups=16, in_channels=64, out_channels=64)
        #
        # self.mst2 = MinimumSpanningTree(TreeFilter2D.norm2_distance, 32,  3)
        # self.tree_filter2 = TreeFilter2D(groups=8, in_channels=32, out_channels=32)

        self.D0 = EncoderB(1, channels[3], channels[3], downsample=False)  # scale: 1/16
        self.D1 = DecoderB(1, channels[3] + channels[4], channels[4])  # scale: 1/8
        self.D2 = DecoderB(1, channels[4] + channels[5], channels[5])  # scale: 1/4
        self.D3 = DecoderB(1, channels[5], channels[6])  # scale: 1/2
        self.D4 = DecoderB(1, channels[6], channels[6])  # scale: 1

        # self.slice = Slice()
        # self.guide = GuideNN()

        # regression
        self.confidence = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 3, 1, 1, bias=False),
            BatchNorm2d(channels[-1], momentum=BN_MOMENTUM),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels[-1], 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.disp = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 3, 1, 1, bias=False),
            BatchNorm2d(channels[-1], momentum=BN_MOMENTUM),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels[-1], 1, 3, 1, 1, bias=False)
        )
        # self.coeff = nn.Sequential(
        #     nn.Conv2d(channels[5], channels[5], 3, 1, 1, bias=False),
        #     BatchNorm2d(channels[5], momentum=BN_MOMENTUM),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(channels[5], 1, 3, 1, 1, bias=False)
        # )
        # self.conv_mask = nn.Sequential(
        #     nn.Conv2d(channels[5], channels[5], 3, 1, 1, bias=False),
        #     BatchNorm2d(channels[5], momentum=BN_MOMENTUM),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(channels[5], 9*16, kernel_size=(3, 3), stride=(1, 1), padding=1,
        #     dilation=(1, 1), bias=False))

    def guidance_1x(self, mask, disp):
        b, c, h, w = disp.shape
        disp = F.unfold(4 * disp, [3, 3], padding=1).view(b, 1, 9, 1, 1, h, w)
        mask = mask.view(b, 1, 9, 4, h, 4, w).permute(0, 1, 2, 3, 5, 4, 6)
        mask = F.softmax(mask, dim=2)
        up_disp = torch.sum(mask * disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(b, 1, 4 * h, 4 * w)

    def guidance_4x(self, mask, disp):
        b, c, h, w = disp.shape
        disp = F.unfold(4 * disp, [3, 3], padding=1).view(b, 1, 9, 1, 1, h, w)
        mask = mask.view(b, 1, 9, 4, 4, h, w)
        mask = F.softmax(mask, dim=2)
        up_disp = torch.sum(mask * disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(b, 1, 4 * h, 4 * w)

    def forward(self, fea, disp_ini, image=None):
        # scale the input disparity
        start_time = time.time()
        # guide = self.guide(image)  # [B,1/2,H,W]

        disp_ini = disp_ini / (2 ** 5)
        fea_E0 = self.E0(torch.cat((disp_ini, fea), 1))  # scale: 1/4

        fea_E1 = self.E1(fea_E0)  # scale: 1/8
        # fea_E1, tree2 = self.E1(fea_E0), self.mst2(fea_E0)  # scale: 1/8
        # fea_E2, tree1 = self.E2(fea_E1), self.mst1(fea_E1)  # scale: 1/16
        fea_E2 = self.E2(fea_E1)  # scale: 1/16
        fea_D0 = self.D0(fea_E2)  # scale: 1/16

        # print('fea_encoder:%.5f'%(time.time()-start_time))

        fea_D1 = self.D1(torch.cat((self.upsample(fea_D0), fea_E1), 1))  # scale: 1/8
        # fea_D1 = self.tree_filter1(fea_D1, fea_E1, tree1)

        fea_D2 = self.D2(torch.cat((self.upsample(fea_D1), fea_E0), 1))  # scale: 1/4
        # fea_D2 = self.tree_filter2(fea_D2, fea_E0, tree2)

        fea_D3 = self.D3(self.upsample(fea_D2))                          # scale: 1/2
        fea_D4 = self.D4(self.upsample(fea_D3))                          # scale: 1

        # print('fea_decoder:%.5f'%(time.time()-start_time))

        # regression
        confidence = self.confidence(fea_D4)
        disp_res = self.disp(fea_D4)
        # disp_ini = self.guidance_1x(self.conv_mask(fea_D4), disp_ini)
        disp_res = torch.clamp(disp_res, 0)

        disp = F.interpolate(disp_ini, scale_factor=4, mode='bilinear') * (1 - confidence) + disp_res * confidence
        return disp * 2 ** 7


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True,
                 mdconv=False):
        super(Conv2x, self).__init__()
        self.concat = concat

        if deconv and is_3d:
            kernel = (3, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel,
                               stride=2, padding=1)

        if self.concat:
            # if mdconv:
                # self.conv2 = DeformConv2d(out_channels * 2, out_channels, kernel_size=3, stride=1)
            # else:
            self.conv2 = BasicConv(out_channels * 2, out_channels, False, is_3d, bn, relu, kernel_size=3,
                                       stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
                                   padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        assert (x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x

def conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation,
                                   bias=False, groups=groups),
                         BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                         nn.LeakyReLU(0.1, inplace=True))



class HourglassRefinement(SubModule):
    """Height and width need to be divided by 16"""
    def __init__(self):
        super(HourglassRefinement, self).__init__()

        # Left and warped error
        in_channels = 4
        self.conv1 = conv2d(in_channels, 16, 7, 1, 3)
        self.conv2 = conv2d(1, 16)  # on low disparity

        self.conv_start = conv2d(32, 32)

        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)

        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)
        self.conv4b = Conv2x(96, 128)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

        # self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)
        self.confidence = nn.Sequential(
            convbn(32, 32, 3, 1, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

        self.disp = nn.Sequential(
            convbn(32, 32, 3, 1, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1, bias=False)
        )


    def forward(self, disp_ini, left_img, right_img):
        warped_feature_left = warp_right_to_left(right_img, disp_ini)
        error = channel_length((left_img - warped_feature_left).contiguous())
        concat1 = torch.cat((error, left_img), dim=1)  # [B, 6, H, W]
        conv1 = self.conv1(concat1)  # [B, 16, H, W]
        conv2 = self.conv2(disp_ini)  # [B, 16, H, W]
        x = torch.cat((conv1, conv2), dim=1)  # [B, 32, H, W]

        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x

        x = self.deconv4a(x, rem3)
        x = self.deconv3a(x, rem2)
        x = self.deconv2a(x, rem1)
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)

        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)  # [B, 32, H, W]

        disp_res = self.disp(x)
        disp_res = torch.clamp(disp_res, 0)

        confidence = self.confidence(x)
        disp_final = disp_ini * (1 - confidence) + disp_res * confidence
        return disp_final


class RefinementNet_global(SubModule):
    def __init__(self, input_channel, channels):
        super(RefinementNet_global, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.downsample = nn.AvgPool2d(2)
        self.global_encoder = GlobalEncoder(96)
        self.mha_weight = nn.Sequential(
            nn.Conv2d(96, 96, 1)
        )
        self.mha_scale = nn.Parameter(torch.FloatTensor([1]))
        self.multihead_attn = nn.MultiheadAttention(96, 4, dropout=0.1)

        self.E0 = EncoderB(1, input_channel, channels[0], downsample=False)  # scale: 1/4
        self.E1 = EncoderB(1, channels[0], channels[1], downsample=True)  # scale: 1/8
        self.E2 = EncoderB(1, channels[1], channels[2], downsample=True)  # scale: 1/16

        self.D0 = EncoderB(1, channels[3], channels[3], downsample=False)  # scale: 1/16
        self.D1 = DecoderB(1, channels[3] + channels[4], channels[4])  # scale: 1/8
        self.D2 = DecoderB(1, channels[4] + channels[5], channels[5])  # scale: 1/4
        self.D3 = DecoderB(1, channels[5], channels[6])  # scale: 1/2
        self.D4 = DecoderB(1, channels[6], channels[6])  # scale: 1

        # regression
        self.confidence = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 3, 1, 1, bias=False),
            BatchNorm2d(channels[-1], momentum=BN_MOMENTUM),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels[-1], 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.disp = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 3, 1, 1, bias=False),
            BatchNorm2d(channels[-1], momentum=BN_MOMENTUM),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels[-1], 1, 3, 1, 1, bias=False)
        )

    def global_branch(self, x):
        x = self.global_encoder.layer1(F.max_pool2d(x, (2, 2)))
        x = self.global_encoder.layer2(F.max_pool2d(x, (2, 2)))
        return x

    def forward(self, fea, disp_ini, image=None):
        # scale the input disparity
        disp_ini = disp_ini / (2 ** 5)

        fea_E0 = self.E0(torch.cat((disp_ini, fea), 1))  # scale: 1/4

        fea_E1 = self.E1(fea_E0)  # scale: 1/8
        fea_E2 = self.E2(fea_E1)  # scale: 1/16

        b, c, h, w = fea_E2.shape
        flatten_x = fea_E2.flatten(2).permute(2, 0, 1)

        global_features = self.global_branch(fea_E2)
        global_flatten = global_features.flatten(2).permute(2, 0, 1)

        tgt = self.multihead_attn(query=flatten_x,
                                  key=global_flatten,
                                  value=global_flatten)[0]

        fea_E2 = fea_E2 + F.interpolate(global_features, scale_factor=4, mode="bilinear") + \
                 self.mha_scale * self.mha_weight(tgt.permute(1, 2, 0).view(b, c, h, w))

        fea_D0 = self.D0(fea_E2)  # scale: 1/16
        fea_D1 = self.D1(torch.cat((self.upsample(fea_D0), fea_E1), 1))  # scale: 1/8
        fea_D2 = self.D2(torch.cat((self.upsample(fea_D1), fea_E0), 1))  # scale: 1/4
        fea_D3 = self.D3(self.upsample(fea_D2))                          # scale: 1/2
        fea_D4 = self.D4(self.upsample(fea_D3))                          # scale: 1

        # regression
        confidence = self.confidence(fea_D4)
        disp_res = self.disp(fea_D4)
        disp_res = torch.clamp(disp_res, 0)

        disp = F.interpolate(disp_ini, scale_factor=4, mode='bilinear') * (1 - confidence) + disp_res * confidence

        # scale the output disparity
        # note that, the size of output disparity is 4 times larger than the input disparity
        return disp * 2 ** 7


class RefinementNet_residual(SubModule):
    def __init__(self, input_channel, channels):
        super(RefinementNet_residual, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.downsample = nn.AvgPool2d(2)

        self.E0 = EncoderB(1, input_channel, channels[0], downsample=False)  # scale: 1/4
        self.E1 = EncoderB(1, channels[0], channels[1], downsample=True)  # scale: 1/8
        self.E2 = EncoderB(1, channels[1], channels[2], downsample=True)  # scale: 1/16

        self.D0 = EncoderB(1, channels[3], channels[3], downsample=False)  # scale: 1/16
        self.D1 = DecoderB(1, channels[3] + channels[4], channels[4])  # scale: 1/8
        self.D2 = DecoderB(1, channels[4] + channels[5], channels[5])  # scale: 1/4
        self.D3 = DecoderB(1, channels[5], channels[6])  # scale: 1/2
        self.D4 = DecoderB(1, channels[6], channels[6])  # scale: 1

        self.confidence = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 3, 1, 1, bias=False),
            BatchNorm2d(channels[-1], momentum=BN_MOMENTUM),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels[-1], 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )
        # regression
        # self.confidence2 = ConfidenceConv(channels[5])
        # self.confidence3 = ConfidenceConv(channels[6])
        # self.confidence4 = ConfidenceConv(channels[6])

        self.disp2 = DispConv(channels[5])
        self.disp3 = DispConv(channels[6])
        self.disp4 = DispConv(channels[6])

    def global_branch(self, x):

        x = self.global_encoder.layer1(F.max_pool2d(x, (2, 2)))
        x = self.global_encoder.layer2(F.max_pool2d(x, (2, 2)))
        return x

    def forward(self, fea, disp_ini, image=None):
        # scale the input disparity
        disp_ini = disp_ini / (2 ** 5)

        fea_E0 = self.E0(torch.cat((disp_ini, fea), 1))  # scale: 1/4

        fea_E1 = self.E1(fea_E0)  # scale: 1/8
        fea_E2 = self.E2(fea_E1)  # scale: 1/16

        fea_D0 = self.D0(fea_E2)  # scale: 1/16
        fea_D1 = self.D1(torch.cat((self.upsample(fea_D0), fea_E1), 1))  # scale: 1/8
        fea_D2 = self.D2(torch.cat((self.upsample(fea_D1), fea_E0), 1))  # scale: 1/4

        residual2 = self.disp2(fea_D2)
        disp2 = disp_ini + residual2                                        # scale: 1/4
        disp2 = self.upsample(disp2)                                        # scale: 1/2
        fea_D3 = self.D3(self.upsample(fea_D2))                             # scale: 1/2

        residual3 = self.disp3(fea_D3)
        disp3 = disp2 + residual3                                           # scale: 1/2
        disp3 = self.upsample(disp3)                                        # scale: 1
        fea_D4 = self.D4(self.upsample(fea_D3))                             # scale: 1

        residual4 = self.disp4(fea_D4)
        confidence = self.confidence(fea_D4)
        disp_final = disp3 * (1 - confidence) + residual4 * confidence   # scale: 1

        # scale the output disparity
        # note that, the size of output disparity is 4 times larger than the input disparity
        return disp_final * 2 ** 7

# class RefinementNet(SubModule):
#     def __init__(self, input_channel, channels):
#         super(RefinementNet, self).__init__()
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         self.downsample = nn.AvgPool2d(2)
#
#         self.E0 = EncoderB(1, input_channel, channels[0], downsample=False)  # scale: 1/4
#         self.E1 = EncoderB(1, channels[0], channels[1], downsample=True)  # scale: 1/8
#         self.E2 = EncoderB(1, channels[1], channels[2], downsample=True)  # scale: 1/16
#
#         self.D0 = EncoderB(1, channels[3], channels[3], downsample=False)  # scale: 1/16
#         self.D1 = DecoderB(1, channels[3] + channels[4], channels[4])  # scale: 1/8
#         self.D2 = DecoderB(1, channels[4] + channels[5], channels[5])  # scale: 1/4
#         self.D3 = DecoderB(1, channels[5], channels[6])  # scale: 1/2
#         self.D4 = DecoderB(1, channels[6], channels[6])  # scale: 1
#
#         # regression
#         self.confidence = nn.Sequential(
#             nn.Conv2d(channels[-1], channels[-1], 3, 1, 1, bias=False),
#             BatchNorm2d(channels[-1], momentum=BN_MOMENTUM),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(channels[-1], 1, 3, 1, 1, bias=False),
#             nn.Sigmoid()
#         )
#         self.disp = nn.Sequential(
#             nn.Conv2d(channels[-1], channels[-1], 3, 1, 1, bias=False),
#             BatchNorm2d(channels[-1], momentum=BN_MOMENTUM),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(channels[-1], 1, 3, 1, 1, bias=False)
#         )
#
#     def forward(self, fea, disp_ini):
#         # scale the input disparity
#         disp_ini = disp_ini / (2 ** 5)
#
#         fea_E0 = self.E0(torch.cat((disp_ini, fea), 1))  # scale: 1/4
#
#         fea_E1 = self.E1(fea_E0)  # scale: 1/8
#         fea_E2 = self.E2(fea_E1)  # scale: 1/16
#
#         fea_D0 = self.D0(fea_E2)  # scale: 1/16
#         fea_D1 = self.D1(torch.cat((self.upsample(fea_D0), fea_E1), 1))  # scale: 1/8
#         fea_D2 = self.D2(torch.cat((self.upsample(fea_D1), fea_E0), 1))  # scale: 1/4
#         fea_D3 = self.D3(self.upsample(fea_D2))                          # scale: 1/2
#         fea_D4 = self.D4(self.upsample(fea_D3))                          # scale: 1
#
#         # regression
#         confidence = self.confidence(fea_D4)
#         disp_res = self.disp(fea_D4)
#         disp_res = torch.clamp(disp_res, 0)
#
#         disp = F.interpolate(disp_ini, scale_factor=4, mode='bilinear') * (1 - confidence) + disp_res * confidence
#
#         # scale the output disparity
#         # note that, the size of output disparity is 4 times larger than the input disparity
#         return disp * 2 ** 7


# class Guidance_new(nn.Module):
#     def __init__(self, output_dim=64, norm_fn='batch'):
#         super(Guidance_new, self).__init__()
#         self.norm_fn = norm_fn
#
#         if self.norm_fn == 'group':
#             self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
#
#         elif self.norm_fn == 'batch':
#             self.norm1 = BatchNorm2d(32, momentum=BN_MOMENTUM)
#
#         elif self.norm_fn == 'instance':
#             self.norm1 = nn.InstanceNorm2d(32)
#
#         elif self.norm_fn == 'none':
#             self.norm1 = nn.Sequential()
#
#         ## 4x use ##
#         self.conv_start = nn.Sequential(nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3), self.norm1, nn.ReLU(inplace=True))
#         ## 2x use ##
#         # self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
#         # self.relu1 = nn.ReLU(inplace=True)
#
#         self.in_planes = 32
#         self.layer1 = self._make_layer(32, stride=1)
#         self.layer2 = self._make_layer(64, stride=2)
#         self.layer3 = self._make_layer(96, stride=2)
#
#         # output convolution
#         self.conv_g0_a = BasicConv(64, 64, kernel_size=3, padding=1)
#         self.guidance0_a = nn.Conv2d(64, output_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         self.conv_g1_a = BasicConv(64, 64, kernel_size=3, padding=1)
#         self.guidance1_a = nn.Conv2d(64, output_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         self.conv_g2_a = BasicConv(64, 64, kernel_size=3, padding=1)
#         self.guidance2_a = nn.Conv2d(64, output_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         self.conv_g3_a = BasicConv(64, 64, kernel_size=3, stride=(1, 1), padding=1)
#         self.guidance3_a = nn.Conv2d(64, output_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#
#         self.conv_g0_b = BasicConv(96, 96, kernel_size=3, padding=1)
#         self.guidance0_b = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         self.conv_g1_b = BasicConv(96, 96, kernel_size=3, padding=1)
#         self.guidance1_b = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         self.conv_g2_b = BasicConv(96, 96, kernel_size=3, padding=1)
#         self.guidance2_b = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         self.conv_g3_b = BasicConv(96, 96, kernel_size=3, stride=(1, 1), padding=1)
#         self.guidance3_b = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
#                 if m.weight is not None:
#                     nn.init.constant_(m.weight, 1)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#
#     def _make_layer(self, dim, stride=1):
#         layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
#         layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
#         layers = (layer1, layer2)
#
#         self.in_planes = dim
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv_start(x)
#         x = self.layer1(x)
#         x_2 = self.layer2(x)
#         x_3 = self.layer3(x_2)
#
#         x2 = self.conv_g0_a(x_2)
#         g0_a = self.guidance0_a(x2)
#         x2 = self.conv_g1_a(x2)
#         g1_a = self.guidance1_a(x2)
#         x2 = self.conv_g2_a(x2)
#         g2_a = self.guidance2_a(x2)
#         x2 = self.conv_g3_a(x2)
#         g3_a = self.guidance3_a(x2)
#
#         x3 = self.conv_g0_b(x_3)
#         g0_b = self.guidance0_b(x3)
#         x3 = self.conv_g1_b(x3)
#         g1_b = self.guidance1_b(x3)
#         x3 = self.conv_g2_b(x3)
#         g2_b = self.guidance2_b(x3)
#         x3 = self.conv_g3_b(x3)
#         g3_b = self.guidance3_b(x3)
#
#         return dict([('g0_2x', g0_a), ('g1_2x', g1_a), ('g2_2x', g2_a), ('g3_2x', g3_a),
#                      ('g0_4x', g0_b), ('g1_4x', g1_b), ('g2_4x', g2_b), ('g3_4x', g3_b)
#                      ])


class Guidance(nn.Module):
    def __init__(self, output_dim=64, norm_fn='batch'):
        super(Guidance, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)

        elif self.norm_fn == 'batch':
            self.norm1 = BatchNorm2d(32, momentum=BN_MOMENTUM)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        ## 4x use ##
        self.conv_start = nn.Sequential(nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3), self.norm1, nn.ReLU(inplace=True))

        ## 2x use ##
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        # self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        # output convolution
        self.guidance0 = nn.Conv2d(64, output_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_start(x)

        x = self.layer1(x)
        x = self.layer2(x)
        g = self.guidance0(x)

        return g




class CrossAttentionBlock(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels, share_key_query=False,
                query_downsample=None, key_downsample=None, key_query_num_convs=2, value_out_num_convs=1, key_query_norm=True,
                value_out_norm=True, matmul_norm=True, with_out_project=True, **kwargs):
        super(CrossAttentionBlock, self).__init__()
        # norm_cfg, act_cfg = kwargs['norm_cfg'], kwargs['act_cfg']
        # key project
        self.key_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs,
            use_norm=key_query_norm,
        )
        # query project
        if share_key_query:
            assert key_in_channels == query_in_channels
            self.query_project = self.key_project
        else:
            self.query_project = self.buildproject(
                in_channels=query_in_channels,
                out_channels=transform_channels,
                num_convs=key_query_num_convs,
                use_norm=key_query_norm,
            )
        # value project
        self.value_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels if with_out_project else out_channels,
            num_convs=value_out_num_convs,
            use_norm=value_out_norm,
        )
        # out project
        self.out_project = None
        if with_out_project:
            self.out_project = self.buildproject(
                in_channels=transform_channels,
                out_channels=out_channels,
                num_convs=value_out_num_convs,
                use_norm=value_out_norm,
            )

        # downsample
        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm
        self.transform_channels = transform_channels

    '''forward'''
    def forward(self, query_feats, key_feats):
        # query_feats: [batch, channels, height ,width]
        batch_size, channels, height, width = query_feats.shape
        hw = height*width
        query = self.query_project(query_feats)
        if self.query_downsample is not None: query = self.query_downsample(query)
        #query: b, h, hc, d, h*w
        query = query.reshape(batch_size, channels, hw)
        # query = query.permute(0, 3, 2, 1).contiguous()  # batch, h*w,  channels
        query = query.permute(0, 2, 1).contiguous()  # batch, h*w, channels

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)

        key = key.reshape(batch_size, channels, hw)  # b, c, h*w
        value = value.reshape(batch_size, channels, hw)
        value = value.permute(0, 2, 1)  # batch, h*w, c

        sim_map = torch.matmul(query, key)  # batch, h*w, h*w

        if self.matmul_norm:
            sim_map = (channels ** -0.5) * sim_map

        sim_map = F.softmax(sim_map, dim=-1)    # batch, h*w, h*w
        context = torch.matmul(sim_map, value)  # batch, h*w, c
        context = context.permute(0, 2, 1).contiguous()  # batch, c, h*w
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])  # batch, channels, h, w
        if self.out_project is not None:
            context = self.out_project(context)

        return context + query_feats

    '''build project'''
    def buildproject(self, in_channels, out_channels, num_convs, use_norm):
        if use_norm:
            convs = [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                    nn.LeakyReLU(0.1, inplace=True)
                )
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                        BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                        nn.LeakyReLU(0.1, inplace=True)
                    )
                )
        else:
            convs = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)]
            for _ in range(num_convs - 1):
                convs.append(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
                )
        if len(convs) > 1: return nn.Sequential(*convs)
        return convs[0]


# class ParallaxAttentionBlock(nn.Module):
#     def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels, share_key_query=False,
#                 query_downsample=None, key_downsample=None, key_query_num_convs=2, value_out_num_convs=1, key_query_norm=True,
#                 value_out_norm=True, matmul_norm=True, with_out_project=True, **kwargs):
#         super(ParallaxAttentionBlock, self).__init__()
#         # norm_cfg, act_cfg = kwargs['norm_cfg'], kwargs['act_cfg']
#         # key project
#         self.key_project = self.buildproject(
#             in_channels=key_in_channels,
#             out_channels=transform_channels,
#             num_convs=key_query_num_convs,
#             use_norm=key_query_norm,
#         )
#         # query project
#         if share_key_query:
#             assert key_in_channels == query_in_channels
#             self.query_project = self.key_project
#         else:
#             self.query_project = self.buildproject(
#                 in_channels=query_in_channels,
#                 out_channels=transform_channels,
#                 num_convs=key_query_num_convs,
#                 use_norm=key_query_norm,
#             )
#         # value project
#         self.value_project = self.buildproject(
#             in_channels=key_in_channels,
#             out_channels=transform_channels if with_out_project else out_channels,
#             num_convs=value_out_num_convs,
#             use_norm=value_out_norm,
#         )
#         # out project
#         self.out_project = None
#         if with_out_project:
#             self.out_project = self.buildproject(
#                 in_channels=transform_channels,
#                 out_channels=out_channels,
#                 num_convs=value_out_num_convs,
#                 use_norm=value_out_norm,
#             )

#         # downsample
#         self.query_downsample = query_downsample
#         self.key_downsample = key_downsample
#         self.matmul_norm = matmul_norm
#         self.transform_channels = transform_channels
#         # self.fusion = nn.Sequential(convbn_3d(key_in_channels*2, key_in_channels, 3, 1, 1),
#         #                             nn.ReLU(inplace=True),
#         #                             convbn_3d(key_in_channels, key_in_channels, 3, 1, 1))

#     '''forward'''
#     def forward(self, query_feats, key_feats):
#         # query_feats: [batch, channels, disparity, height ,width]
#         head_dim = 8
#         batch_size, channels, disparity, height, width = query_feats.shape
#         query = self.query_project(query_feats)
#         if self.query_downsample is not None: query = self.query_downsample(query)
#         #query: b, h, hc, d, h*w
#         query = query.reshape(batch_size, channels*disparity, height, width)
#         # query = query.permute(0, 3, 2, 1).contiguous()  # batch, h*w, disparity, channels
#         query = query.permute(0, 1, 3, 2).contiguous()  # batch, channels, width, height

#         key = self.key_project(key_feats)
#         value = self.value_project(key_feats)
#         if self.key_downsample is not None:
#             key = self.key_downsample(key)
#             value = self.key_downsample(value)

#         key = key.reshape(batch_size, channels*disparity, height, width)
#         value = value.reshape(batch_size, channels*disparity, height, width)
#         value = value.permute(0, 1, 3, 2)  # batch, channels*disparity, width, height

#         sim_map = torch.matmul(query, key)  # batch, channels*d, width, width

#         if self.matmul_norm:
#             sim_map = (head_dim ** -0.5) * sim_map

#         sim_map = F.softmax(sim_map, dim=-1)    # batch, channels*d, width, width
#         context = torch.matmul(sim_map, value)  # batch, channels*d, width, height
#         context = context.permute(0, 1, 3, 2).contiguous()
#         context = context.reshape(batch_size, channels, disparity, height, width)         # batch, channels, disparity, height, width

#         if self.out_project is not None:
#             context = self.out_project(context)

#         return query_feats + context

#     '''build project'''
#     def buildproject(self, in_channels, out_channels, num_convs, use_norm):
#         if use_norm:
#             convs = [
#                 nn.Sequential(
#                     nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
#                     BatchNorm3d(out_channels, momentum=BN_MOMENTUM),
#                     nn.LeakyReLU(0.1, inplace=True)
#                 )
#             ]
#             for _ in range(num_convs - 1):
#                 convs.append(
#                     nn.Sequential(
#                         nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
#                         BatchNorm3d(out_channels, momentum=BN_MOMENTUM),
#                         nn.LeakyReLU(0.1, inplace=True)
#                     )
#                 )
#         else:
#             convs = [nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)]
#             for _ in range(num_convs - 1):
#                 convs.append(
#                     nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
#                 )
#         if len(convs) > 1: return nn.Sequential(*convs)
#         return convs[0]

# class DisparityAttentionBlock(nn.Module):
#     def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels, share_key_query=False,
#                 query_downsample=None, key_downsample=None, key_query_num_convs=2, value_out_num_convs=1, key_query_norm=True,
#                 value_out_norm=True, matmul_norm=True, with_out_project=True, **kwargs):
#         super(DisparityAttentionBlock, self).__init__()
#         # norm_cfg, act_cfg = kwargs['norm_cfg'], kwargs['act_cfg']
#         # key project
#         self.key_project = self.buildproject(
#             in_channels=key_in_channels,
#             out_channels=transform_channels,
#             num_convs=key_query_num_convs,
#             use_norm=key_query_norm,
#         )
#         # query project
#         if share_key_query:
#             assert key_in_channels == query_in_channels
#             self.query_project = self.key_project
#         else:
#             self.query_project = self.buildproject(
#                 in_channels=query_in_channels,
#                 out_channels=transform_channels,
#                 num_convs=key_query_num_convs,
#                 use_norm=key_query_norm,
#             )
#         # value project
#         self.value_project = self.buildproject(
#             in_channels=key_in_channels,
#             out_channels=transform_channels if with_out_project else out_channels,
#             num_convs=value_out_num_convs,
#             use_norm=value_out_norm,
#         )
#         # out project
#         self.out_project = None
#         if with_out_project:
#             self.out_project = self.buildproject(
#                 in_channels=transform_channels,
#                 out_channels=out_channels,
#                 num_convs=value_out_num_convs,
#                 use_norm=value_out_norm,
#             )
#
#         # downsample
#         self.query_downsample = query_downsample
#         self.key_downsample = key_downsample
#         self.matmul_norm = matmul_norm
#         self.transform_channels = transform_channels
#         # self.fusion = nn.Sequential(convbn_3d(key_in_channels*2, key_in_channels, 3, 1, 1),
#         #                             nn.ReLU(inplace=True),
#         #                             convbn_3d(key_in_channels, key_in_channels, 3, 1, 1))
#
#     '''forward'''
#     def forward(self, query_feats, key_feats):
#         # query_feats: [batch, channels, disparity, height ,width]
#         head_dim = 8
#         batch_size, channels, disparity, height, width = query_feats.shape
#         hw = height*width
#         query = self.query_project(query_feats)
#         if self.query_downsample is not None: query = self.query_downsample(query)
#         #query: b, h, hc, d, h*w
#         query = query.reshape(batch_size, channels//head_dim, head_dim, disparity, hw)
#         # query = query.permute(0, 3, 2, 1).contiguous()  # batch, h*w, disparity, channels
#         query = query.permute(0, 4, 1, 3, 2).contiguous()  # batch, h*w, head, disparity, head_c
#
#         key = self.key_project(key_feats)
#         value = self.value_project(key_feats)
#         if self.key_downsample is not None:
#             key = self.key_downsample(key)
#             value = self.key_downsample(value)
#
#         key = key.reshape(batch_size, channels//head_dim, head_dim, disparity, hw)
#         key = key.permute(0, 4, 1, 2, 3)  # batch, h*w, head, head_c, disparity
#         value = value.reshape(batch_size, channels//head_dim, head_dim, disparity, hw)
#         value = value.permute(0, 4, 1, 3, 2)  # batch, h*w, head, disparity, head_c
#
#         sim_map = torch.matmul(query, key)  # batch, h*w, head, d, d
#
#         if self.matmul_norm:
#             sim_map = (head_dim ** -0.5) * sim_map
#
#         sim_map = F.softmax(sim_map, dim=-1)    # batch, h*w, head, d, d
#         context = torch.matmul(sim_map, value)  # batch, h*w, head, disparity, head_c
#         context = context.permute(0, 1, 2, 4, 3).flatten(2, 3)         # batch, h*w, channels, disparity
#
#         context = context.permute(0, 2, 3, 1).contiguous()  # batch, channels, d, h*w
#         context = context.reshape(batch_size, -1, *query_feats.shape[2:])  # batch, channels, d, h, w
#         if self.out_project is not None:
#             context = self.out_project(context)
#
#         context = self.fusion(torch.cat([query_feats, context], dim=1))
#         return query_feats + context
#
#     '''build project'''
#     def buildproject(self, in_channels, out_channels, num_convs, use_norm):
#         if use_norm:
#             convs = [
#                 nn.Sequential(
#                     nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
#                     BatchNorm3d(out_channels, momentum=BN_MOMENTUM),
#                     nn.LeakyReLU(0.1, inplace=True)
#                 )
#             ]
#             for _ in range(num_convs - 1):
#                 convs.append(
#                     nn.Sequential(
#                         nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
#                         BatchNorm3d(out_channels, momentum=BN_MOMENTUM),
#                         nn.LeakyReLU(0.1, inplace=True)
#                     )
#                 )
#         else:
#             convs = [nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)]
#             for _ in range(num_convs - 1):
#                 convs.append(
#                     nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
#                 )
#         if len(convs) > 1: return nn.Sequential(*convs)
#         return convs[0]


# class ParallaxAttentionBlock(nn.Module):
#     def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels, share_key_query=False,
#                  query_downsample=None, key_downsample=None, key_query_num_convs=2, value_out_num_convs=1, key_query_norm=True,
#                  value_out_norm=True, matmul_norm=True, with_out_project=True, **kwargs):
#         super(ParallaxAttentionBlock, self).__init__()
#         # norm_cfg, act_cfg = kwargs['norm_cfg'], kwargs['act_cfg']
#         # key project
#         self.key_project = self.buildproject(
#             in_channels=key_in_channels,
#             out_channels=transform_channels,
#             num_convs=key_query_num_convs,
#             use_norm=key_query_norm,
#         )
#         # query project
#         if share_key_query:
#             assert key_in_channels == query_in_channels
#             self.query_project = self.key_project
#         else:
#             self.query_project = self.buildproject(
#                 in_channels=query_in_channels,
#                 out_channels=transform_channels,
#                 num_convs=key_query_num_convs,
#                 use_norm=key_query_norm,
#             )
#         # value project
#         self.value_project = self.buildproject(
#             in_channels=key_in_channels,
#             out_channels=transform_channels if with_out_project else out_channels,
#             num_convs=value_out_num_convs,
#             use_norm=value_out_norm,
#         )
#         # out project
#         self.out_project = None
#         if with_out_project:
#             self.out_project = self.buildproject(
#                 in_channels=transform_channels,
#                 out_channels=out_channels,
#                 num_convs=value_out_num_convs,
#                 use_norm=value_out_norm,
#             )
#
#         # downsample
#         self.query_downsample = query_downsample
#         self.key_downsample = key_downsample
#         self.matmul_norm = matmul_norm
#         self.transform_channels = transform_channels
#
#     '''forward'''
#     def forward(self, query_feats, key_feats):
#         # query_feats: [batch, channels, disparity, height ,width]
#         head_dim = 8
#         batch_size, channels, height, width = query_feats.shape
#         query = self.query_project(query_feats)
#         if self.query_downsample is not None: query = self.query_downsample(query)
#         query = query.reshape(-1, head_dim, height, width)
#         query = query.permute(0, 2, 3, 1).contiguous()  # batch*c, h, w, c//d
#
#         key = self.key_project(key_feats)
#         value = self.value_project(key_feats)
#         if self.key_downsample is not None:
#             key = self.key_downsample(key)
#             value = self.key_downsample(value)
#
#         key = key.reshape(-1, head_dim, height, width)
#         key = key.permute(0, 2, 1, 3)  # batch*c, h, c//d, w
#         value = value.reshape(-1, head_dim, height, width)
#         value = value.permute(0, 2, 3, 1)  # batch*c, h, w, c//d
#
#         sim_map = torch.matmul(query, key)  # batch*c, h, w, w
#
#         if self.matmul_norm:
#             sim_map = (head_dim ** -0.5) * sim_map
#
#         sim_map = F.softmax(sim_map, dim=-1)    # batch*c, h, w, w
#         context = torch.matmul(sim_map, value)  # batch*c, h, w, c//d
#         context = context.reshape(batch_size, -1, height, width, head_dim)
#         context = context.permute(0, 1, 4, 2, 3).flatten(1, 2).contiguous()         # batch, c, h, w
#
#         if self.out_project is not None:
#             context = self.out_project(context)
#         return context
#
    # '''build project'''
    # def buildproject(self, in_channels, out_channels, num_convs, use_norm):
    #     if use_norm:
    #         convs = [
    #             nn.Sequential(
    #                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
    #                 BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
    #                 nn.LeakyReLU(0.1, inplace=True)
    #             )
    #         ]
    #         for _ in range(num_convs - 1):
    #             convs.append(
    #                 nn.Sequential(
    #                     nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
    #                     BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
    #                     nn.LeakyReLU(0.1, inplace=True)
    #                 )
    #             )
    #     else:
    #         convs = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)]
    #         for _ in range(num_convs - 1):
    #             convs.append(
    #                 nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
    #             )
    #     if len(convs) > 1: return nn.Sequential(*convs)
    #     return convs[0]
