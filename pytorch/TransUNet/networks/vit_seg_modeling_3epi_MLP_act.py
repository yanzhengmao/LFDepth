# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin
from re import X

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
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, query, key, value):
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2).contiguous())
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, 1536)
        self.fc_1 = Linear(1536,  1536)
        self.fc2 = Linear(1536, config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc_1(x)
        x = self.act_fn(x)
        x = self.fc_1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)


        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = grid_size
            patch_size_real = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
            num_patches = patch_size_real[0] * patch_size_real[1]
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        
        self.position_encoder = nn.Sequential(
            nn.Conv2d(1, config.hidden_size*4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(config.hidden_size*4, config.hidden_size, kernel_size=1, stride=1, padding=0),
        )
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size), requires_grad=False)
        
        self.layer_1 = nn.Conv2d(1, 16, 2, 1)
        self.BN1 = nn.BatchNorm2d(16)

        self.layer_2 = nn.Conv2d(16, 32, 2, 1)
        self.BN2 = nn.BatchNorm2d(32)

        self.layer_3 = nn.Conv2d(32, 64, 2, 1)
        self.BN3 = nn. BatchNorm2d(64)

        self.layer_4 = nn.Conv2d(64, 128, 2, 1)
        self.BN4 = nn.BatchNorm2d(128)

        self.layer_5 = nn.Conv2d(128, 256, 2, 1)
        self.BN5 = nn.BatchNorm2d(256)

        self.layer_6 = nn.Conv2d(256, 384, 2, 1)
        self.BN6 = nn.BatchNorm2d(384)

        self.layer_7 = nn.Conv2d(384, 512, 2, 1)
        self.BN7 = nn.BatchNorm2d(512)
        
        self.layer_8 = nn.Conv2d(512, 512, 2, 1)
        self.BN8 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU()

        self.patch_embeddings = Conv2d(in_channels=512,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)

        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.softmax = nn.Softmax(dim=-1)
    
    # def initialize_weights(self):
    #     # initialization
    #     # initialize (and freeze) pos_embed by sin-cos embedding
    #     pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int((self.pos_embed.shape[1]-1)**.5), cls_token=True)
    #     self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    #     # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
    #     w = self.patch_embeddings.weight.data
    #     torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    #     # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
    #     torch.nn.init.normal_(self.cls_token, std=.02)

    #     # initialize nn.Linear and nn.LayerNorm
    #     self.apply(self._init_weights)
    
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         # we use xavier_uniform following official JAX ViT:
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
            
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward(self, x, mask_ratio, batch_size, num_dir):
        srcH, srcW = x.shape[2:]
        
        #(2,2)卷积
        x = self.relu(self.BN1(self.layer_1(x)))
        x = self.relu(self.BN2(self.layer_2(x)))
        x = self.relu(self.BN3(self.layer_3(x)))
        x = self.relu(self.BN4(self.layer_4(x)))
        x = self.relu(self.BN5(self.layer_5(x)))
        x = self.relu(self.BN6(self.layer_6(x)))
        x = self.relu(self.BN7(self.layer_7(x)))
        x = self.relu(self.BN8(self.layer_8(x)))
   
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        # x += self.position_embedding(x, srcW)
        x = x.flatten(2)
        x = x.transpose(-1, -2).contiguous()  # (B, n_patches, hidden)
        b, n, _ = x.shape
        x = x + self.position_embeddings[:, 1:, :]
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)
        mask = None
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        num_epis = b // (batch_size * num_dir)
        x = x.view(batch_size, num_epis * num_dir * n, -1)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # # dist_token = repeat(self.dist_token, '() n d -> b n d', b=b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.position_embeddings

        # x = x.flatten(2)

        embeddings = x
        embeddings = self.dropout(embeddings)

        return embeddings, mask
        # return embeddings, features
    # def position_embedding(self, x, srcW):
    #     B, N, h, w = x.shape
    #     coords_w = torch.arange(0, w, device=x.device).float() * srcW
    #     coords_w = coords_w.view(1,1,1,w).repeat(B,1,h,1)
    #     coords_w = self.position_encoder(coords_w)
    #     return coords_w

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)
        self.layer = nn.Conv2d(config.hidden_size, config.hidden_size, 1, 1)

    def forward(self, query, key, value):
        h = query
        x = self.attention_norm(query)
        x, weights = self.attn(query=x, key=x, value=x)
        x = x + h

        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(query=x, key=key , value=value)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        # cla, att = x[:, 0], x[:, 1:]
        # B, C, N = att.shape
        # att = att.reshape((B, 2, 10, -1)).permute(0, 3, 1, 2).contiguous()
        # att += res
        # att = self.layer(att)

        # x = torch.cat([cla.unsqueeze(dim=1), att.reshape((B, 768, -1)).permute(0, 2, 1).contiguous()], dim=1)
        return x, weights

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
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            # nn.LayerNorm(config.hidden_size),

            nn.Linear(config.hidden_size, config.num_classes)
            )
        self.dist_mlp_head = nn.Sequential(
            # nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.num_classes)
            )

    def forward(self, hidden_states):
        attn_weights = []
        B, C, N = hidden_states.shape
        # res_block = torch.zeros([B, N, 2, 10]).cuda()
        query = hidden_states[:, 0, ...].unsqueeze(1)
        key = hidden_states[:, 1:, ...]
        for layer_block in self.layer:
            # hidden_states, res_block, weights = layer_block(hidden_states, res_block)
            query, weights = layer_block(query, key, key)
            if self.vis:
                attn_weights.append(weights)
        # encoded = self.encoder_norm(hidden_states)
        # x = encoded[:, 0]
        query = self.encoder_norm(query.squeeze())

        # x = self.to_latent(x)
        # x = self.mlp_head(x)
        x = self.to_latent(query)
        x = self.mlp_head(x)
        # x_dist = self.dist_mlp_head(x_dist)

        return x, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids, mask_ratio, batch_size, num_dir):
        embedding_output, mask = self.embeddings(input_ids, mask_ratio, batch_size, num_dir)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, mask


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

        x = hidden_states.permute(0, 2, 1).contiguous()
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

    def forward(self, x, mask_ratio, batch_size, num_dir):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        x0, attn_weights, mask = self.transformer(x, mask_ratio, batch_size, num_dir)  # (B, n_patch, hidden)
        return x0

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            # self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            # posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            # posemb_new = self.transformer.embeddings.position_embeddings
            # if posemb.size() == posemb_new.size():
            #     self.transformer.embeddings.position_embeddings.copy_(posemb)
            # elif posemb.size()[1]-1 == posemb_new.size()[1]:
            #     posemb = posemb[:, 1:]
            #     self.transformer.embeddings.position_embeddings.copy_(posemb)
            # else:
            #     logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
            #     ntok_new = posemb_new.size(1)
            #     if self.classifier == "seg":
            #         _, posemb_grid = posemb[:, :1], posemb[0, 1:]
            #     gs_old = int(np.sqrt(len(posemb_grid)))
            #     gs_new = int(np.sqrt(ntok_new))
            #     print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
            #     posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
            #     zoom = (gs_new / gs_old, gs_new / gs_old, 1)
            #     posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
            #     posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
            #     posemb = posemb_grid

            #     # self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))
            #     self.transformer.embeddings.position_embeddings[:,0:posemb.shape[1],:]=np2th(posemb)
            # Encoder whole
            # for bname, block in self.transformer.encoder.named_children():
            #     for uname, unit in block.named_children():
            #         print(uname, unit)
                    # unit.load_from(weights, n_block=uname)

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