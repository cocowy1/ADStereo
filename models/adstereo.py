from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math
import time
import matplotlib.pyplot as plt
import matplotlib.image as image
from models.lib.nn import SynchronizedBatchNorm2d, SynchronizedBatchNorm3d

BatchNorm2d = SynchronizedBatchNorm2d
BatchNorm3d = SynchronizedBatchNorm3d
BN_MOMENTUM = 0.1


class feature_extraction(nn.Module):
    def __init__(self, concat_feature_channel=16):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.first_conv = nn.Sequential(convbn(3, 32, 7, 2, 3, 1),
                                        nn.ReLU(inplace=True),
                                        convbn(32, 32, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True),
                                        convbn(32, 32, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 3, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 3)

        self.guidance = nn.Sequential(convbn(320, 64, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1,
                                                bias=False))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        b, c, h, w = x.shape
        batch_dim = b // 2

        x = self.first_conv(x)
        x = self.layer1(x)
        low_features = x
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        last_feature = torch.cat((l2, l3, l4), dim=1)
        concat_feature = self.lastconv(last_feature)
        guidance = self.guidance(last_feature)

        guidance_l, _ = torch.split(guidance, [batch_dim, batch_dim], dim=0)

        return guidance_l, last_feature, concat_feature


class ALignBlock(nn.Module):
    def __init__(self, in_channel):
        super(ALignBlock, self).__init__()
        self.conv_start = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3,
                                                  stride=1, padding=1, bias=False),
                                        BatchNorm2d(in_channel, momentum=BN_MOMENTUM),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channel, 2, kernel_size=3,
                                                  stride=1, padding=1, bias=False))

    def flow_warp(self, disp, flow):
        b, _, h, w = disp.shape

        xx = torch.arange(0, w, device=flow.device, dtype=disp.dtype)
        yy = torch.arange(0, h, device=flow.device, dtype=disp.dtype)

        xx = xx.view(1, -1).repeat(h, 1)
        yy = yy.view(-1, 1).repeat(1, w)

        xx = xx.view(1, 1, h, w).repeat(b, 1, 1, 1)
        yy = yy.view(1, 1, h, w).repeat(b, 1, 1, 1)

        xx = xx + flow[:, :1, :, :]
        yy = yy + flow[:, 1:, :, :]
        xx = 2.0 * xx / max(w - 1, 1) - 1.0
        yy = 2.0 * yy / max(h - 1, 1) - 1.0

        grid = torch.cat((xx, yy), 1)
        vgrid = grid.permute(0, 2, 3, 1).contiguous()

        output = F.grid_sample(disp.float(), vgrid.float(), align_corners=True)
        return output

    def get_color_bar(self, color_bar):
        b, c, h, w = color_bar.shape
        xx = torch.arange(0, w, device=color_bar.device, dtype=color_bar.dtype)
        yy = torch.arange(0, h, device=color_bar.device, dtype=color_bar.dtype)

        xx = xx.view(1, -1).repeat(h, 1)
        yy = yy.view(-1, 1).repeat(1, w)

        xx = xx.view(1, 1, h, w).repeat(b, 1, 1, 1)
        yy = yy.view(1, 1, h, w).repeat(b, 1, 1, 1)

        xx = 2.0 * xx / max(w - 1, 1) - 1.0
        yy = 2.0 * yy / max(h - 1, 1) - 1.0

        grid = torch.cat((yy,xx), 1)
        vgrid = grid.squeeze(1)
        return vgrid

    def forward(self, x, disp):
        b, _, h, w = x.shape
        flow = self.conv_start(x)
        disp_final = self.flow_warp(disp, 4 * flow)
        # self.flow_vis(flow, disp, disp_final)
        return disp_final


class AlignModule(SubModule):
    """Height and width need to be divided by 16"""

    def __init__(self):
        super(AlignModule, self).__init__()

        in_channels = 3
        self.conv1 = conv2d(in_channels, 8, 7, 1, 3)
        self.conv2 = conv2d(4, 8, 7, 1, 3)

        self.conv_start = conv2d(16, 16, 5, 1, 2)

        self.conv1a = BasicConv(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        # self.conv4a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        # # #
        # self.deconv4a = Conv2x(96, 64, deconv=True)
        self.deconv3a = Conv2x(64, 48, deconv=True)
        self.deconv2a = Conv2x(48, 32, deconv=True)
        self.deconv1a = Conv2x(32, 16, deconv=True)

        self.flow_make = ALignBlock(16)

    def forward(self, disp_ini, left_img, right_img):
        left_feature = self.conv1(left_img)
        right_feature = self.conv1(right_img)
        warped_feature_left = warp_right_to_left(right_feature, disp_ini)
        error = channel_length((left_feature - warped_feature_left).contiguous())
        concat1 = torch.cat((error, left_img), dim=1)  # [B, 4, H, W]
        # conv1 = self.conv1(concat1)  # [B, 8, H, W]
        conv2 = self.conv2(concat1)  # [B, 8, H, W]
        x = torch.cat((left_feature, conv2), dim=1)  # [B, 16, H, W]

        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        # x = self.conv4a(x)
        #
        # x = self.deconv4a(x, rem3)
        x = self.deconv3a(x, rem2)
        x = self.deconv2a(x, rem1)
        x = self.deconv1a(x, rem0)

        disp_final = self.flow_make(x, disp_ini)

        return disp_final


class PropgationNet_1x(nn.Module):
    def __init__(self, input_channel):
        super(PropgationNet_1x, self).__init__()
        self.mask = nn.Sequential(convbn(input_channel, input_channel * 2, 3, 2, 1, 1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(input_channel * 2, 9 * 64, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                            dilation=(1, 1), bias=False))

    def forward(self, guidance, disp):
        b, c, h, w = disp.shape
        disp = F.unfold(8 * disp, [3, 3], padding=1).view(b, 1, 9, 1, 1, h, w)
        mask = self.mask(guidance).view(b, 1, 9, 8, 8, h, w)
        mask = F.softmax(mask, dim=2)
        up_disp = torch.sum(mask * disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(b, 1, 8 * h, 8 * w)


class UpsamplingModule(nn.Module):
    def __init__(self, in_channels):
        super(UpsamplingModule, self).__init__()
        self.deconv = nn.Sequential(
            Deconv2dUnit(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1,
                         bn=False, relu=True, bn_momentum=0.01),
            Conv2dUnit(in_channels, in_channels, 3, stride=1, padding=1, relu=False, bn=True)
        )

    def forward(self, cur_fea, pre_fea):
        pre_fea = self.deconv(pre_fea)
        structure = (cur_fea - pre_fea).abs()
        return structure, cur_fea, pre_fea


class Adaptive_downsample(nn.Module):
    def __init__(self, input_channel, kernel, scale, padding, use_group):
        super(Adaptive_downsample, self).__init__()
        self.use_group = use_group

        self.kernel = kernel
        self.out_channel = kernel ** 2
        self.downsample_scale = scale
        self.padding = padding

        if self.use_group:
            num_groups = 8
            self.groups = input_channel // num_groups
        else:
            self.groups = input_channel

        self.convmask = nn.Sequential(
            convbn(input_channel, input_channel * self.downsample_scale, self.kernel, self.downsample_scale,
                   self.padding, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(input_channel * self.downsample_scale, self.groups * self.out_channel, kernel_size=1,
                      stride=(1, 1), padding=0,
                      dilation=(1, 1), bias=False))

    def forward(self, x):
        b, c, h, w = x.shape
        rem = x
        mask = self.convmask(rem).view(b, 1, self.groups, self.out_channel, h // self.downsample_scale,
                                       w // self.downsample_scale)  # b,g,9,h//2,w//2

        if self.use_group:
            rem_down = F.unfold(rem, [self.kernel, self.kernel], stride=(self.downsample_scale, self.downsample_scale),
                                padding=(self.padding, self.padding)).view(b, c // self.groups, self.groups,
                                                                           self.out_channel, h // self.downsample_scale,
                                                                           w // self.downsample_scale)
            mask = F.softmax(mask, dim=3)
            downsample_x = torch.sum(mask * rem_down, dim=3)

        else:
            rem_down = F.unfold(rem, [self.kernel, self.kernel], stride=(self.downsample_scale, self.downsample_scale),
                                padding=(self.padding, self.padding)).view(b, 1, c, self.out_channel,
                                                                           h // self.downsample_scale,
                                                                           w // self.downsample_scale)
            mask = F.softmax(mask, dim=3)
            downsample_x = torch.sum(mask * rem_down, dim=3)

        downsample_x = downsample_x.reshape(b, -1, h // self.downsample_scale, w // self.downsample_scale)

        return downsample_x


class Pre_processing(nn.Module):
    def __init__(self, gw_channels, cat_channels, maxdisp, gruops, kernel, scale, padding, use_structure=False):
        super(Pre_processing, self).__init__()
        if self.training and use_structure:
            self.use_structure = use_structure
        else:
            self.use_structure = False

        self.gw_channels = gw_channels
        self.cat_channels = cat_channels
        self.maxdisp = maxdisp
        self.num_groups = gruops

        self.kernel = kernel
        self.downsample_scale = scale
        self.padding = padding

        self.gwc_ds = Adaptive_downsample(self.gw_channels, self.kernel, self.downsample_scale, self.padding, True)
        self.cat_ds = Adaptive_downsample(self.cat_channels, self.kernel, self.downsample_scale, self.padding, False)

        if self.use_structure:
            self.loss_detection_gwc = UpsamplingModule(self.gw_channels)
            self.loss_detection_cat = UpsamplingModule(self.cat_channels)
        
    def forward(self, gwc_feature, concat_feature):
        g_8x = self.gwc_ds(gwc_feature)
        c_8x = self.cat_ds(concat_feature)
        b, _, _, _ = gwc_feature.shape
        l_g_8x, r_g_8x = torch.split(g_8x, [b//2, b//2], dim=0)
        l_c_8x, r_c_8x = torch.split(c_8x, [b//2, b//2], dim=0)

        if self.use_structure:
            structure_gwc, cur_fea_gwc, pre_fea_gwc = self.loss_detection_gwc(gwc_feature, g_8x)
            structure_cat, cur_fea_cat, pre_fea_act = self.loss_detection_cat(concat_feature, c_8x)

            structure_list = [structure_gwc, structure_cat]

        ### build gwc volume ###
        g = build_gwc_volume(l_g_8x, r_g_8x, self.maxdisp // (4 * self.downsample_scale), self.num_groups)
        c = build_concat_volume(l_c_8x, r_c_8x, self.maxdisp // (4 * self.downsample_scale))
        volume = torch.cat((g, c), 1)

        if self.training and self.use_structure:
            return volume, structure_list
        else:
            return volume


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            BatchNorm3d(in_channels * 2, momentum=BN_MOMENTUM))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            BatchNorm3d(in_channels, momentum=BN_MOMENTUM))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


class ADStereo(nn.Module):
    def __init__(self, maxdisp, mixed_precision, use_structure=False, refine=False):
        super(ADStereo, self).__init__()
        self.maxdisp = maxdisp
        self.num_groups = 40
        self.mixed_precision = mixed_precision
        self.use_structure = use_structure
        self.refine = refine
        
        self.kernel = 3
        self.downsample_scale = 2
        self.padding = 1

        self.concat_channels = 12
        self.feature_extraction = feature_extraction(concat_feature_channel=self.concat_channels)

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)
        self.dres3 = hourglass(32)
        self.dres4 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.pre_processing = Pre_processing(320, 12, self.maxdisp, self.num_groups, self.kernel, self.downsample_scale,
                                             self.padding, self.use_structure)

        self.align = AlignModule()
        self.up = PropgationNet_1x(64)
        self.refine = refine

        # self.top_k = 2
        # middlebury
        self.top_k = 2
        if self.refine:
            self.refinement = HourglassRefinement()

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
            # elif isinstance(m, nn.Linear):
            #     m.bias.data.zero_()


    def topkpool(self, cost, k):
        if k == 0:
            _, ind = cost.sort(1, True)
            pool_ind_ = ind[:, :, :k]
            b, _, h, w = pool_ind_.shape
            pool_ind = pool_ind_.new_zeros((b, 3, h, w))
            pool_ind[:, 1:2] = pool_ind_
            pool_ind[:, 0:1] = torch.max(
                pool_ind_ - 1, pool_ind_.new_zeros(pool_ind_.shape))
            pool_ind[:, 2:] = torch.min(
                pool_ind_ + 1, self.D * pool_ind_.new_ones(pool_ind_.shape))
            cv = torch.gather(cost, 2, pool_ind)

            disp = pool_ind

        else:
            _, ind = cost.sort(1, True)
            pool_ind = ind[:, :k, ...]
            cv = torch.gather(cost, 1, pool_ind)
            disp = pool_ind

        return cv, disp

    def forward(self, left, right):
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.autocast(device_type="cuda", enabled=self.mixed_precision):
            guidance, gwc_features, concat_features = self.feature_extraction(
                torch.cat([left, right], dim=0))

            if self.training and self.use_structure:
                volume, structure_list = self.pre_processing(gwc_features, concat_features)
            else:
                volume = self.pre_processing(gwc_features, concat_features)

            cost0 = self.dres0(volume)
            cost0 = self.dres1(cost0) + cost0

            out1 = self.dres2(cost0) + cost0
            out2 = self.dres3(out1) + cost0
            out3 = self.dres4(out2) + cost0

            cost3 = self.classif3(out3).squeeze(1)
            cost3_topk, disp = self.topkpool(cost3, self.top_k)
            prob3 = F.softmax(cost3_topk, dim=1)
            pred3_8x = torch.sum(prob3 * disp, 1, keepdim=True)
            #
            pred3_1x = self.up(guidance, pred3_8x)
            pred4 = self.align(pred3_1x, left, right)

            if self.refine:
                pred5 = self.refine(pred4, left, right)
                
            if self.training:
                outputs = {}
        
                cost0 = self.classif0(cost0)
                cost1 = self.classif1(out1)
                cost2 = self.classif2(out2)

                cost0 = F.interpolate(cost0, scale_factor=(
                    4 * self.downsample_scale, 4 * self.downsample_scale, 4 * self.downsample_scale), mode='trilinear')
                cost1 = F.interpolate(cost1, scale_factor=(
                    4 * self.downsample_scale, 4 * self.downsample_scale, 4 * self.downsample_scale), mode='trilinear')
                cost2 = F.interpolate(cost2, scale_factor=(
                    4 * self.downsample_scale, 4 * self.downsample_scale, 4 * self.downsample_scale), mode='trilinear')

                cost0 = torch.squeeze(cost0, 1)
                pred0 = F.softmax(cost0, dim=1)
                pred0_1x = disparity_regression(pred0, self.maxdisp)

                cost1 = torch.squeeze(cost1, 1)
                pred1 = F.softmax(cost1, dim=1)
                pred1_1x = disparity_regression(pred1, self.maxdisp)

                cost2 = torch.squeeze(cost2, 1)
                pred2 = F.softmax(cost2, dim=1)
                pred2_1x = disparity_regression(pred2, self.maxdisp)

                if self.use_structure:
                    outputs["structure_discrepancy"] = structure_list
                    
                if self.refine:
                    outputs["disp"] = [pred0_1x, pred1_1x, pred2_1x, pred3_1x, pred4, pred5]
                    return outputs
                else:
                    outputs["disp"] = [pred0_1x, pred1_1x, pred2_1x, pred3_1x, pred4]
                    return outputs

            else:
                torch.cuda.synchronize()
                # print('total time:%.4f'%(time.time()-start_time))
                return pred4, pred3_1x
