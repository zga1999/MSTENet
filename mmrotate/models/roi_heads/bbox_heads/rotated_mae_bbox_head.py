# Copyright (c) OpenMMLab. All rights reserved.
"""MAE decoder used as the RoI bbox head of imTED.

Ported from the mmrotate 0.x
``mmrotate/models/roi_heads/bbox_heads/rotated_mae_bbox_head.py``. Only the
framework API was adapted (registry, checkpoint helpers, logger, base class);
the decoder architecture and the parameter names are unchanged so that MAE
pretrained checkpoints still load.
"""
import math
import os
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from mmdet.models.roi_heads.bbox_heads import BBoxHead
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import _load_checkpoint
from timm.layers import trunc_normal_
from torch.utils.checkpoint import checkpoint

from mmrotate.registry import MODELS
from ...backbones.vit import Block


@MODELS.register_module()
class RotatedMAEBBoxHead(BBoxHead):
    """Rotated MAE decoder bbox head.

    The RoI feature map is flattened into tokens, projected to ``embed_dim``
    and refined by ``depth`` MAE decoder blocks before the classification and
    regression fcs.

    Args:
        in_channels (int): Channels of the input RoI feature map.
        img_size (int): Image size the fixed sin-cos position embedding is
            built for. Defaults to 224.
        patch_size (int): Patch size of the ViT backbone. Defaults to 16.
        embed_dim (int): Width of the decoder. Defaults to 256.
        depth (int): Number of decoder blocks. Defaults to 4.
        num_heads (int): Number of attention heads. Defaults to 8.
        mlp_ratio (float): Ratio of the mlp hidden dim. Defaults to 4.
        use_checkpoint (bool): Use gradient checkpointing in the decoder
            blocks. Defaults to False.
        rois_mode (str, optional): Kept for interface compatibility with
            :class:`RotatedMAEBBoxHeadSTDC`; ``None`` means the head does not
            consume ``rois``. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 img_size=224,
                 patch_size=16,
                 embed_dim=256,
                 depth=4,
                 num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 use_checkpoint=False,
                 init_cfg=None,
                 rois_mode=None,
                 **kwargs):
        super(RotatedMAEBBoxHead, self).__init__(
            in_channels=in_channels, init_cfg=init_cfg, **kwargs)

        if init_cfg is None:
            self.init_cfg = None
        self.rois_mode = rois_mode

        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        num_patches = (img_size // patch_size)**2
        self.num_patches = num_patches

        # MAE decoder specifics
        self.norm = norm_layer(in_channels)
        self.decoder_embed = nn.Linear(in_channels, embed_dim, bias=True)
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.decoder_blocks = nn.ModuleList([
            Block(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer) for i in range(depth)
        ])
        # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.decoder_box_norm = norm_layer(embed_dim)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(embed_dim, self.num_classes + 1)
        if self.with_reg:
            # XYWHA
            out_dim_reg = (5 if self.reg_class_agnostic else
                           5 * self.num_classes)
            self.fc_reg = nn.Linear(embed_dim, out_dim_reg)

    def _init_weights(self, m):
        """Initialize a single module and refresh the sin-cos pos embed."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches**.5),
            cls_token=True)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

    def init_weights(self):
        """Load MAE decoder weights that match this head; rest stay random."""
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            self.apply(self._init_weights)
            return

        assert 'checkpoint' in self.init_cfg, \
            f'Only support specify `Pretrained` in `init_cfg` in ' \
            f'{self.__class__.__name__} '
        pretrained = self.init_cfg['checkpoint']
        if not os.path.isfile(pretrained):
            raise ValueError(f'checkpoint path {pretrained} is invalid')

        self.apply(self._init_weights)

        logger.info(f'loading MAE decoder for {self.__class__.__name__}')
        ckpt = _load_checkpoint(pretrained, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt

        skip_exact = {
            'pos_embed', 'cls_token', 'mask_token',
            'decoder_norm.weight', 'decoder_norm.bias',
            'decoder_pred.weight', 'decoder_pred.bias',
        }
        model_sd = self.state_dict()
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('patch_embed') or k.startswith('blocks'):
                continue
            if k in skip_exact:
                continue
            if k in model_sd and model_sd[k].shape == v.shape:
                new_state_dict[k] = v

        incompatible = self.load_state_dict(new_state_dict, strict=False)
        logger.info(
            f'loaded {len(new_state_dict)} MAE decoder tensors '
            f'(skipped {len(incompatible.missing_keys)} detection-only keys)')

    def interpolate_pos_encoding(self, x, w, h):
        """Interpolate the fixed pos embed to the current token grid."""
        npatch = x.shape[1] - 1
        N = self.decoder_pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.decoder_pos_embed
        class_pos_embed = self.decoder_pos_embed[:, 0]
        patch_pos_embed = self.decoder_pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the
        # interpolation, see https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)),
                                    dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(
            h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed),
                         dim=1)

    def forward(self, x):
        """Forward function."""
        B, C, W, H = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.decoder_embed(x)

        x = x + self.interpolate_pos_encoding(x, W * self.patch_size,
                                              H * self.patch_size)[:, 1:, :]
        for blk in self.decoder_blocks:
            if self.use_checkpoint:
                x = checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.decoder_box_norm(x.mean(dim=1))
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py  # noqa: E501
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed],
                                   axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
