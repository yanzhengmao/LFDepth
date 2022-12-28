# Copyright (C) 2020 Guanglei Yang
#
# This file is a part of PGA
# add btsnet with attention .
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

import torch
import torch.nn as nn
import math
from torch.nn.functional import unfold
from torch.nn import functional as F
from AttentionGraphCondKernel import AttentionGraphCondKernel
from TransUNet.networks.vit_seg_modeling_epi_MLP_act import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling_epi_MLP_act import CONFIGS as CONFIGS_ViT_seg
from TransUNet.networks.vit_seg_modeling_epi_MLP_act_patch import VisionTransformer as ViT_seg_patch
from TransUNet.networks.vit_seg_modeling_epi_MLP_act_patch import CONFIGS as CONFIGS_ViT_seg
# from TransUNet.networks.vit_seg_modeling_epi_MLP_act_patch import VisionTransformer_cnn as ViT_seg_cnn
import numpy as np


def bn_init_as_tf(m):
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = True  # These two lines enable using stats (moving mean and var) loaded from pretrained model_Binocular
        m.eval()  # or zero mean and variance of one if the batch norm layer has no pretrained values
        m.affine = True
        m.requires_grad = True


def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class get_accuracy(nn.Module):
    def __init__(self):
        super(get_accuracy, self).__init__()
        self.sotfmax = nn.Softmax(dim=-1)

    def forward(self, output, annotation):
        channel_dim = -1
        soft_conv_t3 = self.sotfmax(output)
        annotation_pred = torch.argmax(soft_conv_t3, dim=channel_dim)
        # correct_pre = annotation.eq_(annotation_pred.float())
        correct_pre = torch.eq(annotation_pred, annotation).float()
        accuracy = torch.mean(correct_pre)
        return annotation_pred, correct_pre, accuracy

class get_mask(nn.Module):
    def __init__(self):
        super(get_mask, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, pred_list, annotation):
        mask = []
        for i in range(len(pred_list)):
            mask_i = torch.ones_like(annotation)
            channel_dim = -1
            soft_conv_t3 = self.softmax(pred_list[i])
            annotation_pred = torch.argmax(soft_conv_t3, dim=channel_dim)
            mask_i = torch.where(abs(annotation_pred - annotation) < 3, mask_i, torch.zeros_like(annotation))
            mask.append(mask_i)
        return mask

class get_output_mask(nn.Module):
    def __init__(self):
        super(get_output_mask, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, output, annotation, accuracy):
        mask = torch.ones_like(annotation)
        channel_dim = -1
        soft_conv_t3 = self.softmax(output)
        annotation_pred = torch.argmax(soft_conv_t3, dim=channel_dim)
        mask = torch.where(abs(annotation_pred - annotation) < min(2.0 / accuracy, 229), mask, torch.zeros_like(annotation))
        return mask    
        

class get_stage2_loss(nn.Module):
    def __init__(self):
        super(get_stage2_loss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, output,  pred_list, tra_hot_annotation, mask):
        fuse_loss = torch.tensor(0., requires_grad=True).cuda()
        ms_wei, mo_wei, com_wei = (1, 1, 1)

        # print(('-- num of muti-scale infer: {} --').format(len(pred_list)))
        for i in range(len(pred_list)):
            fuse_loss += mo_wei * (-(torch.sum(torch.sum(self.logsoftmax(pred_list[i]).mul(tra_hot_annotation), dim=-1) * mask[i]))) / torch.sum(mask[i])
        #
        mask_total = torch.stack([mask[i] for i in range(len(mask))], -1)
        mask_total = torch.clip(torch.sum(mask_total, dim=-1), min=0, max=1)
        # # print(('-- num of final_infer: {} --').format(1))
        fuse_loss += (-(torch.sum(torch.sum(self.logsoftmax(output).mul(tra_hot_annotation), dim=-1) * mask_total))) / torch.sum(mask_total)
        # fuse_loss = (-(torch.sum(self.logsoftmax(pred_patch).mul(tra_hot_annotation)))) / tra_hot_annotation.shape[0]
        # fuse_loss += (-(torch.sum(self.logsoftmax(pred_disp).mul(tra_hot_annotation)))) / tra_hot_annotation.shape[0]
        return fuse_loss
    
class get_mask_loss(nn.Module):
    def __init__(self):
        super(get_mask_loss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, output, tra_hot_annotation, mask):

        # # print(('-- num of final_infer: {} --').format(1))
        fuse_loss = (-(torch.sum(torch.sum(self.logsoftmax(output).mul(tra_hot_annotation), dim=-1) * mask))) / torch.sum(mask)
        # fuse_loss = (-(torch.sum(self.logsoftmax(pred_patch).mul(tra_hot_annotation)))) / tra_hot_annotation.shape[0]
        # fuse_loss += (-(torch.sum(self.logsoftmax(pred_disp).mul(tra_hot_annotation)))) / tra_hot_annotation.shape[0]
        return fuse_loss

class get_stage1_loss(nn.Module):
    def __init__(self):
        super(get_stage1_loss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, output, tra_hot_annotation):
        fuse_loss = torch.tensor(0., requires_grad=True).cuda()
        ms_wei, mo_wei, com_wei = (1, 1, 1)

        # print(('-- num of muti-scale infer: {} --').format(len(pred_list)))
        # for i in range(len(pred_list)):
        #     fuse_loss += mo_wei * (-(torch.sum(self.logsoftmax(pred_list[i]).mul(tra_hot_annotation)))) / \
        #                  tra_hot_annotation.shape[0]
        #
        # # print(('-- num of final_infer: {} --').format(1))
        fuse_loss += (-(torch.sum(self.logsoftmax(output).mul(tra_hot_annotation)))) / \
                     tra_hot_annotation.shape[0]
        # fuse_loss = (-(torch.sum(self.logsoftmax(pred_patch).mul(tra_hot_annotation)))) / tra_hot_annotation.shape[0]
        # fuse_loss += (-(torch.sum(self.logsoftmax(pred_disp).mul(tra_hot_annotation)))) / tra_hot_annotation.shape[0]
        return fuse_loss


class atrous_conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, apply_bn_first=True):
        super(atrous_conv, self).__init__()
        self.atrous_conv = torch.nn.Sequential()
        if apply_bn_first:
            self.atrous_conv.add_module('first_bn', nn.BatchNorm2d(in_channels, momentum=0.01, affine=True,
                                                                   track_running_stats=True, eps=1.1e-5))

        self.atrous_conv.add_module('aconv_sequence', nn.Sequential(nn.ReLU(),
                                                                    nn.Conv2d(in_channels=in_channels,
                                                                              out_channels=out_channels * 2, bias=False,
                                                                              kernel_size=1, stride=1, padding=0),
                                                                    nn.BatchNorm2d(out_channels * 2, momentum=0.01,
                                                                                   affine=True,
                                                                                   track_running_stats=True),
                                                                    nn.ReLU(),
                                                                    nn.Conv2d(in_channels=out_channels * 2,
                                                                              out_channels=out_channels, bias=False,
                                                                              kernel_size=3, stride=1,
                                                                              padding=(dilation, dilation),
                                                                              dilation=dilation)))

    def forward(self, x):
        return self.atrous_conv.forward(x)


class upconv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2):
        super(upconv, self).__init__()
        self.elu = nn.ELU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1,
                              padding=1)
        self.ratio = ratio

    def forward(self, x):
        up_x = F.interpolate(x, scale_factor=self.ratio, mode='nearest')
        out = self.conv(up_x)
        out = self.elu(out)
        return out


class reduction_1x1(nn.Sequential):
    def __init__(self, num_in_filters, num_out_filters, max_depth, is_final=False):
        super(reduction_1x1, self).__init__()
        self.max_depth = max_depth
        self.is_final = is_final
        self.sigmoid = nn.Sigmoid()
        self.reduc = torch.nn.Sequential()

        while num_out_filters >= 4:
            if num_out_filters < 8:
                if self.is_final:
                    self.reduc.add_module('final',
                                          torch.nn.Sequential(nn.Conv2d(num_in_filters, out_channels=1, bias=False,
                                                                        kernel_size=1, stride=1, padding=0),
                                                              nn.Sigmoid()))
                else:
                    self.reduc.add_module('plane_params', torch.nn.Conv2d(num_in_filters, out_channels=3, bias=False,
                                                                          kernel_size=1, stride=1, padding=0))
                break
            else:
                self.reduc.add_module('inter_{}_{}'.format(num_in_filters, num_out_filters),
                                      torch.nn.Sequential(
                                          nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters,
                                                    bias=False, kernel_size=1, stride=1, padding=0),
                                          nn.ELU()))

            num_in_filters = num_out_filters
            num_out_filters = num_out_filters // 2

    def forward(self, net):
        net = self.reduc.forward(net)
        if not self.is_final:
            theta = self.sigmoid(net[:, 0, :, :]) * math.pi / 3
            phi = self.sigmoid(net[:, 1, :, :]) * math.pi * 2
            dist = self.sigmoid(net[:, 2, :, :]) * self.max_depth
            n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
            n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
            n3 = torch.cos(theta).unsqueeze(1)
            n4 = dist.unsqueeze(1)
            net = torch.cat([n1, n2, n3, n4], dim=1)

        return net


class local_planar_guidance(nn.Module):
    def __init__(self, upratio):
        super(local_planar_guidance, self).__init__()
        self.upratio = upratio
        self.u = torch.arange(self.upratio).reshape([1, 1, self.upratio]).float()
        self.v = torch.arange(int(self.upratio)).reshape([1, self.upratio, 1]).float()
        self.upratio = float(upratio)

    def forward(self, plane_eq, focal):
        plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.upratio), 2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded, int(self.upratio), 3)
        n1 = plane_eq_expanded[:, 0, :, :]
        n2 = plane_eq_expanded[:, 1, :, :]
        n3 = plane_eq_expanded[:, 2, :, :]
        n4 = plane_eq_expanded[:, 3, :, :]

        u = self.u.repeat(plane_eq.size(0), plane_eq.size(2) * int(self.upratio), plane_eq.size(3)).cuda()
        u = (u - (self.upratio - 1) * 0.5) / self.upratio

        v = self.v.repeat(plane_eq.size(0), plane_eq.size(2), plane_eq.size(3) * int(self.upratio)).cuda()
        v = (v - (self.upratio - 1) * 0.5) / self.upratio

        return n4 / (n1 * u + n2 * v + n3)


class bts(nn.Module):
    def __init__(self, params, feat_out_channels, num_features=512):
        super(bts, self).__init__()
        self.params = params

        self.upconv5 = upconv(feat_out_channels[4], num_features)
        self.bn5 = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)

        self.conv5 = torch.nn.Sequential(
            nn.Conv2d(num_features + feat_out_channels[3], num_features, 3, 1, 1, bias=False),
            nn.ELU())
        self.upconv4 = upconv(num_features, num_features // 2)
        self.bn4 = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv4 = torch.nn.Sequential(
            nn.Conv2d(num_features // 2 + feat_out_channels[2], num_features // 2, 3, 1, 1, bias=False),
            nn.ELU())
        self.bn4_2 = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)

        self.daspp_3 = atrous_conv(num_features // 2, num_features // 4, 3, apply_bn_first=False)
        self.daspp_6 = atrous_conv(num_features // 2 + num_features // 4 + feat_out_channels[2], num_features // 4, 6)
        self.daspp_12 = atrous_conv(num_features + feat_out_channels[2], num_features // 4, 12)
        self.daspp_18 = atrous_conv(num_features + num_features // 4 + feat_out_channels[2], num_features // 4, 18)
        self.daspp_24 = atrous_conv(num_features + num_features // 2 + feat_out_channels[2], num_features // 4, 24)
        self.daspp_conv = torch.nn.Sequential(
            nn.Conv2d(num_features + num_features // 2 + num_features // 4, num_features // 4, 3, 1, 1, bias=False),
            nn.ELU())
        self.reduc8x8 = reduction_1x1(num_features // 4, num_features // 4, self.params.max_depth)
        self.lpg8x8 = local_planar_guidance(8)

        self.upconv3 = upconv(num_features // 4, num_features // 4)
        self.bn3 = nn.BatchNorm2d(num_features // 4, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv3 = torch.nn.Sequential(
            nn.Conv2d(num_features // 4 + feat_out_channels[1] + 1, num_features // 4, 3, 1, 1, bias=False),
            nn.ELU())
        self.reduc4x4 = reduction_1x1(num_features // 4, num_features // 8, self.params.max_depth)
        self.lpg4x4 = local_planar_guidance(4)

        self.upconv2 = upconv(num_features // 4, num_features // 8)
        self.bn2 = nn.BatchNorm2d(num_features // 8, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(num_features // 8 + feat_out_channels[0] + 1, num_features // 8, 3, 1, 1, bias=False),
            nn.ELU())

        self.reduc2x2 = reduction_1x1(num_features // 8, num_features // 16, self.params.max_depth)
        self.lpg2x2 = local_planar_guidance(2)

        self.upconv1 = upconv(num_features // 8, num_features // 16)
        self.reduc1x1 = reduction_1x1(num_features // 16, num_features // 32, self.params.max_depth, is_final=True)
        self.conv1 = torch.nn.Sequential(nn.Conv2d(num_features // 16 + 4, num_features // 16, 3, 1, 1, bias=False),
                                         nn.ELU())
        self.get_depth = torch.nn.Sequential(nn.Conv2d(num_features // 16, 1, 3, 1, 1, bias=False),
                                             nn.Sigmoid())
        # self.get_depth = torch.nn.Sequential(nn.Conv2d(num_features // 16, 1, 3, 1, 1, bias=False))

        # scale 0

    def forward(self, features, focal):
        skip0, skip1, skip2, skip3 = features[1], features[2], features[3], features[4]
        dense_features = torch.nn.ReLU()(features[5])
        upconv5 = self.upconv5(dense_features)  # H/16
        upconv5 = self.bn5(upconv5)

        concat5 = torch.cat([upconv5, skip3], dim=1)
        iconv5 = self.conv5(concat5)

        upconv4 = self.upconv4(iconv5)  # H/8
        upconv4 = self.bn4(upconv4)
        concat4 = torch.cat([upconv4, skip2], dim=1)
        iconv4 = self.conv4(concat4)
        iconv4 = self.bn4_2(iconv4)

        daspp_3 = self.daspp_3(iconv4)
        concat4_2 = torch.cat([concat4, daspp_3], dim=1)
        daspp_6 = self.daspp_6(concat4_2)
        concat4_3 = torch.cat([concat4_2, daspp_6], dim=1)
        daspp_12 = self.daspp_12(concat4_3)
        concat4_4 = torch.cat([concat4_3, daspp_12], dim=1)
        daspp_18 = self.daspp_18(concat4_4)
        concat4_5 = torch.cat([concat4_4, daspp_18], dim=1)
        daspp_24 = self.daspp_24(concat4_5)
        concat4_daspp = torch.cat([iconv4, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], dim=1)
        daspp_feat = self.daspp_conv(concat4_daspp)

        reduc8x8 = self.reduc8x8(daspp_feat)
        plane_normal_8x8 = reduc8x8[:, :3, :, :]
        plane_normal_8x8 = F.normalize(plane_normal_8x8, 2, 1)
        plane_dist_8x8 = reduc8x8[:, 3, :, :]
        plane_eq_8x8 = torch.cat([plane_normal_8x8, plane_dist_8x8.unsqueeze(1)], 1)
        depth_8x8 = self.lpg8x8(plane_eq_8x8, focal)
        depth_8x8_scaled = depth_8x8.unsqueeze(1) / self.params.max_depth
        depth_8x8_scaled_ds = F.interpolate(depth_8x8_scaled, scale_factor=0.25, mode='nearest')

        upconv3 = self.upconv3(daspp_feat)  # H/4
        upconv3 = self.bn3(upconv3)
        concat3 = torch.cat([upconv3, skip1, depth_8x8_scaled_ds], dim=1)
        iconv3 = self.conv3(concat3)

        reduc4x4 = self.reduc4x4(iconv3)
        plane_normal_4x4 = reduc4x4[:, :3, :, :]
        plane_normal_4x4 = F.normalize(plane_normal_4x4, 2, 1)
        plane_dist_4x4 = reduc4x4[:, 3, :, :]
        plane_eq_4x4 = torch.cat([plane_normal_4x4, plane_dist_4x4.unsqueeze(1)], 1)
        depth_4x4 = self.lpg4x4(plane_eq_4x4, focal)
        depth_4x4_scaled = depth_4x4.unsqueeze(1) / self.params.max_depth
        depth_4x4_scaled_ds = F.interpolate(depth_4x4_scaled, scale_factor=0.5, mode='nearest')

        upconv2 = self.upconv2(iconv3)  # H/2
        upconv2 = self.bn2(upconv2)
        concat2 = torch.cat([upconv2, skip0, depth_4x4_scaled_ds], dim=1)
        iconv2 = self.conv2(concat2)

        reduc2x2 = self.reduc2x2(iconv2)
        plane_normal_2x2 = reduc2x2[:, :3, :, :]
        plane_normal_2x2 = F.normalize(plane_normal_2x2, 2, 1)
        plane_dist_2x2 = reduc2x2[:, 3, :, :]
        plane_eq_2x2 = torch.cat([plane_normal_2x2, plane_dist_2x2.unsqueeze(1)], 1)
        depth_2x2 = self.lpg2x2(plane_eq_2x2, focal)
        depth_2x2_scaled = depth_2x2.unsqueeze(1) / self.params.max_depth

        upconv1 = self.upconv1(iconv2)

        reduc1x1 = self.reduc1x1(upconv1)
        concat1 = torch.cat([upconv1, reduc1x1, depth_2x2_scaled, depth_4x4_scaled, depth_8x8_scaled], dim=1)

        iconv1 = self.conv1(concat1)
        # final_depth = self.params.max_depth * self.get_depth(iconv1)
        final_depth = 8.0 * self.get_depth(iconv1) - 4.0
        # final_depth = self.get_depth(iconv1)

        # if self.params.dataset == 'kitti':
        #     final_depth = final_depth * focal.view(-1, 1, 1, 1).float() / 715.0873

        return depth_8x8_scaled, depth_4x4_scaled, depth_2x2_scaled, reduc1x1, final_depth


class encoder(nn.Module):
    def __init__(self, params):
        super(encoder, self).__init__()
        self.params = params
        import torchvision.models as models
        if params.encoder == 'densenet121_bts':
            self.base_model = models.densenet121(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [64, 64, 128, 256, 1024]
        elif params.encoder == 'densenet161_bts':
            self.base_model = models.densenet161(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [96, 96, 192, 384, 2208]
        elif params.encoder == 'resnet50_bts':
            self.base_model = models.resnet50(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnet101_bts':
            self.base_model = models.resnet101(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnext50_bts':
            self.base_model = models.resnext50_32x4d(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnext101_bts':
            self.base_model = models.resnext101_32x8d(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        else:
            print('Not supported encoder: {}'.format(params.encoder))

    def forward(self, x):
        features = [x]
        skip_feat = [x]
        for k, v in self.base_model._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(features[-1])
            features.append(feature)
            if any(x in k for x in self.feat_names):
                skip_feat.append(feature)

        return skip_feat


class cal_confidence(nn.Module):
    def __int__(self):
        super(cal_confidence, self).__int__()

    def forward(self, tensor_smax):
        max_idxs = torch.argmax(tensor_smax, dim=-1, keepdim=True)

        pad = torch.zeros(size=max_idxs.shape).cuda()
        # (bs, 229+2)
        ext_tensor = torch.cat([pad, tensor_smax, pad], dim=-1)
        # refresh max_idxs
        max_idxs += 1
        # print(max_idxs.size())
        lowers = max_idxs - 1
        uppers = max_idxs + 1

        # ext_tensor (512, 512, 231) vs (512, 512)
        max_probs = torch.gather(ext_tensor, -1, max_idxs).squeeze()
        low_probs = torch.gather(ext_tensor, -1, lowers).squeeze()
        up_probs = torch.gather(ext_tensor, -1, uppers).squeeze()

        # stack_probs: (512, 512, 2) or (bs, 2)
        stack_probs = torch.stack([low_probs, up_probs], dim=-1)
        greater_probs = torch.gather(stack_probs, -1, torch.argmax(stack_probs, dim=-1, keepdim=True)).squeeze()
        # indexing by array just supports boolean mask array, numpy likewise.
        # stack_probs[tf.argmax(stack_probs, axis=-1)]

        # (512, 512) -> (512, 512, 1) or (bs) -> (bs, 1)
        ori_confidence = torch.add(max_probs, greater_probs)
        # confidence = tf.expand_dims(ori_confidence, axis=-1)

        '''
        # repeat confidence, (512, 512, 1) to (512, 512, 229) or (bs, 1) -> (bs, 229)
        class_num = int(tensor_smax.get_shape()[-1])
        # tf.rank() return a tensor
        # use tensor_smax.get_shape().ndims, Not confidence, of which dims is unknown
        repeats = np.ones(confidence.get_shape().ndims - 1, np.uint8).tolist() + [class_num]  # arange : [start, stop)
        rep_confidence = utils.tf_repeat(confidence, repeats)
        '''

        # multiply_confidence
        # return tf.multiply(tensor_smax, confidence, name='mult_confidence') #, ori_confidence

        # -------------------------------------- IMPORTANT --------------------------------------------#
        # gradients for argmax is meaningless, just stop grad computation here, or you would get
        # LookupError: No gradient defined for operation 'ArgMax'
        stop_grad_conf = ori_confidence  # tf.stop_gradient(ori_confidence, name='stop_confidence_grad')
        return stop_grad_conf  # ori_confidence


class get_infer_with_max_conf(nn.Module):
    def __init__(self):
        super(get_infer_with_max_conf, self).__init__()
        self.cal_confidence = cal_confidence()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sub_infer_list, sub_scope):
        # 0 as max
        idx = 0
        max_infer = sub_infer_list[idx]  # base_infer  #
        max_smax = self.softmax(max_infer)  # tf.nn.softmax(max_infer, dim=-1, name='base_infer_smax')  #
        max_conf = self.cal_confidence(max_smax)
        idx += 1

        # conf_thre = tf.Variable(conf_thre_init, trainable=train_thre, name='conf_thre')
        # less_than_thre = tf.less(max_conf, conf_thre, name='less_than_thre')

        for i in range(idx, len(sub_infer_list)):
            sub_infer = sub_infer_list[i]
            sub_smax_i = self.softmax(sub_infer)
            # sub_conf_i : (bs)
            sub_conf_i = self.cal_confidence(sub_smax_i)

            # condition: (bs) mask, sub_infer and max_infer: (bs, 229), condition.shape[0] ==
            # sub_infer.shape[0], then max_infer is still (bs, 229)
            # max_infer = tf.where(tf.logical_and(less_than_thre, tf.greater(sub_conf_i, max_conf)),
            #                     sub_infer, max_infer, name='max_infer')
            # max_infer = torch.where(torch.greater(sub_conf_i, max_conf), sub_infer, max_infer)
            condition = torch.gt(sub_conf_i, max_conf).float()
            diag_list = torch.diag(condition)
            diag_list_y = torch.diag(1 - condition)
            max_infer = torch.add(torch.mm(diag_list, sub_infer), torch.mm(diag_list_y, max_infer))

        return max_infer  # max_infer has not been applied softmax


class pad_2_full_size(nn.Module):
    def __init__(self):
        super(pad_2_full_size, self).__init__()

    def forward(self, source_arr, full_hei, full_wid, prob_distribution=False):
        # assert source_arr.ndim == 2 or source_arr.ndim == 3
        s_hei = source_arr.size()[0]
        s_wid = source_arr.size()[1]
        assert (full_hei - s_hei) % 2 == 0 and (full_wid - s_wid) % 2 == 0
        p_half_hei = (full_hei - s_hei) // 2
        p_half_wid = (full_wid - s_wid) // 2
        full_shape = list(source_arr.size())
        full_shape[0] = full_hei
        full_shape[1] = full_wid
        full_arr = torch.zeros(size=full_shape)
        if prob_distribution:
            full_arr[(Ellipsis, 0)] = 1
        full_arr[p_half_hei:full_hei - p_half_hei, p_half_wid:full_wid - p_half_wid] = source_arr
        return full_arr


class prob2disp(nn.Module):
    def __init__(self):
        super(prob2disp, self).__init__()
        self.label2disp = label2disp()

    def forward(self, prob, ret_confidence=False):
        # prob: (512, 512, 229)
        assert (prob >= 0).all() and prob.ndim == 3
        (hei, wid, class_num) = prob.shape
        assert class_num >= 2

        # tensor_smax:(512, 512, 229) or (bs, 229), MUST BE probability(After softmax)#
        # with tf.variable_scope('confidence'):
        max_idxs = torch.argmax(prob, dim=-1, keepdim=True)

        pad = torch.zeros(size=[max_idxs.shape[0], max_idxs.shape[1]]).unsqueeze(-1).cuda()
        # (bs, 229+2)
        ext_prob = torch.cat([pad, prob, pad], dim=-1)
        # refresh max_idxs
        max_idxs += 1
        lowers = max_idxs - 1
        uppers = max_idxs + 1

        # ext_tensor (512, 512, 231) vs (512, 512)
        max_probs = torch.gather(ext_prob, -1, max_idxs).squeeze()
        low_probs = torch.gather(ext_prob, -1, lowers).squeeze()
        up_probs = torch.gather(ext_prob, -1, uppers).squeeze()

        # stack_probs: (512, 512, 2) or (bs, 2)
        stack_probs = torch.stack([low_probs, up_probs], dim=-1)
        g_idx = torch.argmax(stack_probs, dim=-1, keepdim=True)
        greater_probs = torch.gather(stack_probs, -1, g_idx).squeeze()

        stack_idxs = torch.stack([lowers.squeeze(), uppers.squeeze()], dim=-1)
        greater_neighbour = torch.gather(stack_idxs, -1, g_idx).squeeze()

        # (512, 512) -> (512, 512, 1) or (bs) -> (bs, 1)
        confidence = max_probs + greater_probs

        float_label = (max_probs / confidence) * max_idxs.squeeze() + (greater_probs / confidence) * greater_neighbour
        # MUST - 1, because max_idxs and greater_neighbour within [0, 229+2)
        float_label -= 1

        if ret_confidence:
            return self.label2disp(float_label), confidence
        else:
            return self.label2disp(float_label)


class label2disp(nn.Module):
    def __init__(self):
        super(label2disp, self).__init__()
        self.disp_precition = 0.035
        self.disp_offset = 4

    def forward(self, labels):
        return labels * self.disp_precition - self.disp_offset


class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.in_channels = 229 * 4
        self.out_channels = 229
        self.fc = nn.Linear(self.in_channels, self.out_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.get_infer_with_max_conf = get_infer_with_max_conf()

    def forward(self, pred_list):
        stack_4 = torch.stack(pred_list, dim=1)
        epi_4_smax = self.softmax(stack_4)
        concat = torch.cat(pred_list, dim=-1)
        output = self.fc(concat)
        logits_smax = self.softmax(output)
        infer_4A1_list = [output] + pred_list
        max_disp_infer = self.get_infer_with_max_conf(infer_4A1_list, 'max_disp_infer')
        max_of_4_smax = self.softmax(max_disp_infer)

        return epi_4_smax, logits_smax, max_of_4_smax




class BtsModel(nn.Module):
    def __init__(self, params):
        super(BtsModel, self).__init__()
        # self.encoder = encoder(params)
        ### vit
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 1
        config_vit.n_skip = 3
        config_vit.patches.grid = (1, 1)
        self.in_channels = 229 * 4
        self.out_channels = 229
        self.encoder = ViT_seg(config_vit, img_size=[params.input_height, params.input_width],num_classes=config_vit.n_classes).cuda(params.gpu)
        # self.encoder.load_from(weights=np.load(config_vit.pretrained_path))
        self.fc = nn.Linear(self.in_channels, self.out_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.get_infer_with_max_conf = get_infer_with_max_conf()

    def forward(self, x, focal):
        # pred_list = []
        # for i in range(4):
        #     skip_feat = self.encoder(x[:, i])
        #     pred_list.append(skip_feat)

        # stack_4 = torch.stack(pred_list, dim=1)
        # smax_4 = self.softmax(stack_4)
        # concat = torch.cat(pred_list, dim=-1)
        # output = self.fc(concat)
        
        output = self.encoder(x)

        # epi_patch_cat = torch.cat([pred_patch, output], dim=-1)
        # output_cat = self.fc2(epi_patch_cat)
        # final_cat = torch.cat([output_cat, pred_patch, output], dim=-1)
        # pred_disp = self.fc3(final_cat)
        #
        # pred_list.append(pred_patch)
        # output_smax = self.softmax(output)
        # infer_4A1_list = [output] + pred_list
        # max_disp_infer = self.get_infer_with_max_conf(infer_4A1_list, 'max_disp_infer')
        # max_disp_infer_smax = self.softmax(max_disp_infer)
        # return output, pred_list, smax_4, output_smax, max_disp_infer_smax
        return output

