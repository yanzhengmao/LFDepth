# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}



class Attention(nn.Module):
    """Multi-head Attention"""

    def __init__(self, dim, hidden_dim=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        hidden_dim = hidden_dim or dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qk = nn.Linear(dim, hidden_dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, hidden_dim, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(1. - attn_drop)
        self.proj = nn.Linear(hidden_dim, dim)
        self.proj_drop = nn.Dropout(1. - proj_drop)


    def forward(self, x):
        """Multi-head Attention"""
        B, N, _ = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = self.softmax(torch.matmul(q, k.transpose(-1, -2) * self.scale))
        attn = self.attn_drop(attn)
        x = (torch.matmul(attn, v).permute(0, 2, 1, 3)).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class MLP(nn.Module):
    """MLP"""

    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dropout = nn.Dropout(1. - dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x



class PixelEmbed(nn.Module):
    """Image to Pixel Embedding"""

    def __init__(self, img_size, patch_size=(16,16), in_channels=3, embedding_dim=768, stride=(1,4)):
        super(PixelEmbed, self).__init__()
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  #把384的原图像分成16 * 16的patch, self.num_patches为一共有多少个patch
        new_patch_size = (patch_size[0] // stride[0], patch_size[1] // stride[1])                         #new_patch_size为把分成的patch按4 * 4分为更小的patch，用来pixel embedding
        self.new_patch_size = new_patch_size
        self.inner_dim = embedding_dim // new_patch_size[0] // new_patch_size[1]      #16 * 16的大patch的embedding_dim在分成小patch之后要相对应的进行缩放
        self.proj = nn.Conv2d(in_channels, self.inner_dim, kernel_size=3, padding_mode='pad',
                              padding=1, stride=stride, bias=True)
        self.unfold = _unfold_(kernel_size=new_patch_size)                      #以4 * 4的小patch进行unfold(展开)

    def forward(self, x):
        B = x.shape[0]   #B, 3, 384， 384  大patch尺度为16 * 16共 24 * 24个，每个小patch尺度为4 * 4,共96 * 96个
        x = self.proj(x) # B, C, H, W    在此输入中得到的为  B, 40, 96, 96     piexl encoder 把4 * 4的小patch映射到1 * 40的维度上得到  96 * 96 * 40
        x = self.unfold(x) # B, N, Ck2     返回的维度为 B, N = numH * numW(即4 * 4的小patch在整副图像上的滑动次数) ，Ck2 = C * kernel_size * kernel_size
                         #因为proj(x)得到的为96 * 96个小patch的encoder,每个大patch由4 * 4个小patch组合而成，所以在像素展开时，是按一个大patch进行展开，即kernel_size = 4 * 4
        x = x.reshape(B * self.num_patches, self.inner_dim, -1) # B*N, C, M   #这里要把pixel embedding到整幅图，即每个16 * 16的patch中去
        x = x.permute(0, 2, 1) # B*N, M, C   在此实验中为 [B * 24 * 24, 16, 40]
        return x


class Pixel2Patch(nn.Module):
    """Projecting Pixel Embedding to Patch Embedding"""

    def __init__(self, outer_dim):
        super(Pixel2Patch, self).__init__()
        self.norm_proj = nn.LayerNorm([outer_dim])
        self.proj = nn.Linear(outer_dim, outer_dim)
        self.fake = nn.Parameter(torch.zeros((1, 1, outer_dim)),requires_grad=False)

    def forward(self, pixel_embed, patch_embed):
        B, N, _ = patch_embed.shape                                 #因为patch_embed初始化为 n + 1个patch_embedding，所以这里的N = n + 1
        proj = pixel_embed.reshape(B, N - 1, -1)            #B, 24 * 24, 640
        proj = self.proj(self.norm_proj(proj))                      #B, N, outer_dim
        proj = torch.cat([repeat(self.fake, '() n d -> b n d', b=B), proj], dim=1) #self.title() - > B, 1, outer_dim   proj.shape = [B, N+1, outer_dim]
        patch_embed = patch_embed + proj                            #B， N+1， outer_dim
        return patch_embed


class _unfold_(nn.Module):
    """Unfold"""

    def __init__(self, kernel_size, stride=0):
        super(_unfold_, self).__init__()
        if stride == 0:
            self.stride = kernel_size
        self.kernel_size = kernel_size


    def forward(self, x):
        """TNT"""
        N, C, H, W = x.shape
        numH = int(H / self.kernel_size[0])    #numH为小patch在H上滑动，可以滑动的次数，因为小patch为4 * 4， 所以分别在H, W上除以4, 即为滑动的次数
        numW = int(W / self.kernel_size[1])    #在此实验中 numH = numW = 24
        if numH * self.kernel_size[0] != H or numW * self.kernel_size[1] != W:
            x = x[:, :, :numH * self.kernel_size[0], :, numW * self.kernel_size[1]]
        output_img = x.reshape(N, C, numH, self.kernel_size[0], W)

        output_img = output_img.permute(0, 1, 2, 4, 3)        #N, C, numH, W, self.kernel_size

        output_img = output_img.reshape(N, C, int(
            numH * numW), self.kernel_size[0], self.kernel_size[1])         #N, C, numH * numW, self.kernel_size, self.kernel_size

        output_img = output_img.permute(0, 2, 1, 4, 3)         #N, numH * numW, C, self.kernel_size, self.kernel_size

        output_img = output_img.reshape(N, int(numH * numW), -1) #N, numH * numW, C * self.kernel_size * self.kernel_size
        return output_img

def _get_clones(module, N):
    """get_clones"""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TNTEncoder(nn.Module):
    """TNT"""

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, pixel_embed, patch_embed):
        """TNT"""
        for layer in self.layers:
            pixel_embed, patch_embed = layer(pixel_embed, patch_embed)
        return pixel_embed, patch_embed

class TNT(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3,dropout=0.1,
            attn_dropout=0.1,
            drop_connect=0.1):
        super(TNT, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)
        self.embedding_dims = 768

        grid_size = config.patches["grid"]
        patch_size = grid_size
        patch_size_real = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        num_patches = patch_size_real[0] * patch_size_real[1]
        pixel_patch =(1, 2)
        new_patch_size = (patch_size[0] // pixel_patch[0], patch_size[1] // pixel_patch[1])

        self.relu = nn.ReLU()

        self.layer_1 = nn.Conv2d(in_channels, 16, 1, 1)
        self.layer_2 = nn.Conv2d(16, 32, 2, 1)
        self.BN2 = nn.BatchNorm2d(32)
        self.layer_3 = nn.Conv2d(32, 64, 2, 1)
        self.layer_4 = nn.Conv2d(64, 128, 2, 1)
        self.BN4 = nn.BatchNorm2d(128)
        self.layer_5 = nn.Conv2d(128, 256, 2, 1)
        self.layer_6 = nn.Conv2d(256, 512, 2, 1)
        self.BN6 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()

        inner_dim = self.embedding_dims // new_patch_size[0] // new_patch_size[1]
        self.position_patch_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, self.embedding_dims), requires_grad=True)
        self.position_pixel_embeddings = nn.Parameter(torch.rand(1, inner_dim, new_patch_size[0] * new_patch_size[1]), requires_grad=True)
        self.patch_embeddings = nn.Parameter(torch.zeros((1, num_patches, self.embedding_dims)), requires_grad=False)    # 初始化 n+1 个 patch embedding 来存储模型的特征，它们都初始化为零：
                                                                                         # 其中第一个 patch embedding 又叫 class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size), requires_grad=True)


        self.fake = nn.Parameter(torch.zeros(1, 1, self.embedding_dims), requires_grad=False)
        self.pos_drop = nn.Dropout()

        self.pixel_embed = PixelEmbed(img_size, patch_size, 512, self.embedding_dims, pixel_patch)  # B*N, M, C   B*24*24, 16, 40
        self.pixel2patch = Pixel2Patch(self.embedding_dims)

        inner_config = {'dim': inner_dim, 'num_heads': 4, 'mlp_ratio': 4}
        outer_config = {'dim': self.embedding_dims, 'num_heads': config.transformer.num_heads, 'mlp_ratio': 4}

        encoder_layer = TNTBlock(inner_config, outer_config, dropout=dropout, attn_dropout=attn_dropout,
                                 drop_connect=drop_connect)

        self.encoder = TNTEncoder(encoder_layer, config.transformer.num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm([self.embedding_dims]),
            nn.Linear(self.embedding_dims, config.num_classes)
        )


    def forward(self, x):
        # if self.hybrid:
        #     x, features = self.hybrid_model(x)
        # else:
        #     features = None
        x = self.relu(self.layer_1(x))
        x = self.relu(self.BN2(self.layer_2(x)))
        x = self.relu(self.layer_3(x))
        x = self.relu(self.BN4(self.layer_4(x)))
        x = self.relu((self.layer_5(x)))
        x = self.relu(self.BN6(self.layer_6(x)))

        B, _, _, _ = x.shape
        pixel_embed = self.pixel_embed(x)
        pixel_embed = pixel_embed + self.position_pixel_embeddings.permute(0, 2, 1)  # B*N, M, C

        patch_embed = torch.cat([self.cls_token, self.patch_embeddings], dim=1)  # 初始化 n+1 个 patch embedding 来存储模型的特征，它们都初始化为零：
        # 其中第一个 patch embedding 又叫 class token
        # shape = [1, num_patches + 1, embedding_dim]
        patch_embed = repeat(patch_embed,'() n d -> b n d', b=B)
        patch_embed = self.pos_drop(patch_embed + self.position_patch_embeddings)

        patch_embed = self.pixel2patch(pixel_embed, patch_embed)

        pixel_embed, patch_embed = self.encoder(pixel_embed, patch_embed)

        y = self.head(patch_embed[:, 0])
        return y

    # x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
    #     x = x.flatten(2)
    #     x = x.transpose(-1, -2)  # (B, n_patches, hidden)
    #
    #     b, n, _ = x.shape
    #
    #     cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
    #     x = torch.cat((cls_tokens, x), dim=1)
    #     x += self.position_embeddings[:, :(n + 1)]
    #
    #     # x = x.flatten(2)
    #
    #     embeddings = x
    #     embeddings = self.dropout(embeddings)

        # return embeddings
        # return embeddings, features



class TNTBlock(nn.Module):
    """TNT Block"""

    def __init__(self, inner_config, outer_config, dropout=0., attn_dropout=0., drop_connect=0.):
        super().__init__()
        # inner transformer
        inner_dim = inner_config['dim']
        num_heads = inner_config['num_heads']
        mlp_ratio = inner_config['mlp_ratio']
        self.inner_norm1 = nn.LayerNorm([inner_dim])
        self.inner_attn = Attention(inner_dim, num_heads=num_heads, qkv_bias=True, attn_drop=attn_dropout,
                                    proj_drop=dropout)
        self.inner_norm2 = nn.LayerNorm([inner_dim])
        self.inner_mlp = MLP(inner_dim, int(inner_dim * mlp_ratio), dropout=dropout)
        # outer transformer
        outer_dim = outer_config['dim']
        num_heads = outer_config['num_heads']
        mlp_ratio = outer_config['mlp_ratio']
        self.outer_norm1 = nn.LayerNorm([outer_dim])
        self.outer_attn = Attention(outer_dim, num_heads=num_heads, qkv_bias=True, attn_drop=attn_dropout,
                                    proj_drop=dropout)
        self.outer_norm2 = nn.LayerNorm([outer_dim])
        self.outer_mlp = MLP(outer_dim, int(outer_dim * mlp_ratio), dropout=dropout)
        # pixel2patch
        self.pixel2patch = Pixel2Patch(outer_dim)
        # assistant


    def forward(self, pixel_embed, patch_embed):
        """TNT Block"""
        pixel_embed = pixel_embed + self.inner_attn(self.inner_norm1(pixel_embed))
        pixel_embed = pixel_embed + self.inner_mlp(self.inner_norm2(pixel_embed))

        patch_embed = self.pixel2patch(pixel_embed, patch_embed)

        patch_embed = patch_embed + self.outer_attn(self.outer_norm1(patch_embed))
        patch_embed = patch_embed + self.outer_mlp(self.outer_norm2(patch_embed))
        return pixel_embed, patch_embed


# class Block(nn.Module):
#     def __init__(self, config, vis):
#         super(Block, self).__init__()
#         self.hidden_size = config.hidden_size
#         self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
#         self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
#         self.ffn = self.Mlp(config)
#         self.attn = Attention(config, vis)
#
#     def forward(self, x):
#         h = x
#         x = self.attention_norm(x)
#         x, weights = self.attn(x)
#         x = x + h
#
#         h = x
#         x = self.ffn_norm(x)
#         x = self.ffn(x)
#         x = x + h
#         return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = TNTBlock(config, vis)
            self.layer.append(copy.deepcopy(layer))
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.num_classes))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        x = encoded[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)

        return x, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.encoder = TNT(config, img_size=img_size)
        # self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        output = self.encoder(input_ids)
        # encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return output


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            1024,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.conv_more_ = Conv2dReLU(
            1024,
            2048,
            kernel_size=3,
            stride=2,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        # self.project_f3=Conv2dReLU(
        #     512,
        #     768,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1,
        #     use_batchnorm=True,
        # )

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        # self.transfermer_f34=nn.Transformer(nhead=4, num_encoder_layers=3,d_model=768)
    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        if features[0].shape[2] % 2 == 1:
            h, w = int(features[0].shape[2] / 2)+1, int(features[0].shape[3] / 2)
        else:
            h, w = int(features[0].shape[2]/2), int(features[0].shape[3]/2)
        ### unified multi-scale transformer

        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x1 = self.conv_more(x)
        x2 = self.conv_more_(x1)

        # for i, decoder_block in enumerate(self.blocks):
        #     if features is not None:
        #         skip = features[i] if (i < self.config.n_skip) else None
        #     else:
        #         skip = None
        #     x = decoder_block(x, skip=skip)
        return x1,x2


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        # self.decoder = DecoderCup(config)

        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x0 = self.transformer(x)  # (B, n_patch, hidden)
        return x0

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            # self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid

                # self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))
                self.transformer.embeddings.position_embeddings[:,0:posemb.shape[1],:]=np2th(posemb)
            # Encoder whole
            # for bname, block in self.transformer.encoder.named_children():
            #     for uname, unit in block.named_children():
            #         unit.load_from(weights, n_block=uname)
            #
            # if self.transformer.embeddings.hybrid:
            #     # self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
            #     gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
            #     gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
            #     self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
            #     self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)
            #
            #     for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
            #         for uname, unit in block.named_children():
            #             unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}

