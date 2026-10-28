# Copyright (c) OpenMMLab. All rights reserved.
"""Spatial-Transform-Decoupled (STDC) variant of the MAE RoI bbox head.

Ported from the mmrotate 0.x
``mmrotate/models/roi_heads/bbox_heads/rotated_mae_bbox_head_stdc.py``. Only
the framework API was adapted; the decoder architecture, the staged
``dc``/``am`` schedule and the parameter names are unchanged.

Unlike :class:`RotatedMAEBBoxHead`, the box prediction is not produced by a
final ``fc_reg`` but accumulated across the decoder blocks: each block may
predict a subset of ``(xy, wh, a)``, which the later blocks reuse to build the
attention activation mask.
"""
import math
import os
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from mmdet.models.roi_heads.bbox_heads import BBoxHead
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import _load_checkpoint
from timm.layers import trunc_normal_
from torch.utils.checkpoint import checkpoint

from mmrotate.registry import MODELS
from .activation_mask_stdc import BlockSTDC
from .rotated_mae_bbox_head import get_2d_sincos_pos_embed


@MODELS.register_module()
class RotatedMAEBBoxHeadSTDC(BBoxHead):
    """MAE decoder bbox head with spatial-transform-decoupled regression.

    Args:
        in_channels (int): Channels of the input RoI feature map.
        dc_mode_str_list (list[str]): Per-block decoupled regression mode.
            An entry containing ``'XY'`` / ``'WH'`` / ``'A'`` means that block
            predicts the corresponding box components.
        num_convs_list (list[int]): Per-block number of convs in
            :class:`LayerReg`.
        am_mode_str_list (list[str]): Per-block activation-mask mode, e.g.
            ``'V'`` applies the mask to the attention values.
        rois_mode (str): ``'hbbox'`` or ``'rbbox'``, how ``rois`` are
            interpreted when building the activation mask. Defaults to
            ``'hbbox'``.
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
                 dc_mode_str_list=['', 'XY', 'A', 'WH'],
                 num_convs_list=[0, 3, 2, 1],
                 am_mode_str_list=['', 'V', 'V', 'V'],
                 rois_mode='hbbox',
                 **kwargs):
        super(RotatedMAEBBoxHeadSTDC, self).__init__(
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
            BlockSTDC(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                dc_mode_str=dc_mode_str_list[i],
                have_predicted_str=''.join(dc_mode_str_list[:i]),
                num_convs=num_convs_list[i],
                am_mode_str=am_mode_str_list[i],
                rois_mode=rois_mode,
            ) for i in range(depth)
        ])
        # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.decoder_box_norm = norm_layer(embed_dim)
        # reconstruct fc_cls since input channels are changed; the box
        # prediction is assembled from the decoder blocks, so fc_reg is unused
        if self.with_cls:
            self.fc_cls = nn.Linear(embed_dim, self.num_classes + 1)
        if self.with_reg:
            del self.fc_reg

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

        # detection-only layers (fc_cls / layer_reg / ...) stay randomly inited
        self.apply(self._init_weights)

        logger.info(f'loading MAE decoder for {self.__class__.__name__}')
        ckpt = _load_checkpoint(pretrained, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt

        # skip MAE encoder + MAE-only tokens/reconstruction head
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

        # only intersecting keys → no unexpected/missing spam
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

    def forward(self, x, rois):
        """Forward function.

        Args:
            x (Tensor): RoI features of shape (N, C, W, H).
            rois (Tensor): RoIs of shape (N, 5) or (N, 6), the first column
                being the batch index. Used to build the activation mask.
        """
        B, C, W, H = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.decoder_embed(x)

        x = x + self.interpolate_pos_encoding(x, W * self.patch_size,
                                              H * self.patch_size)[:, 1:, :]

        bbox_pr_xy = torch.zeros([B, 2],
                                 dtype=x.dtype,
                                 device=x.device,
                                 requires_grad=False)
        bbox_pr_wh = torch.zeros([B, 2],
                                 dtype=x.dtype,
                                 device=x.device,
                                 requires_grad=False)
        bbox_pr_a = torch.zeros([B, 1],
                                dtype=x.dtype,
                                device=x.device,
                                requires_grad=False)

        for blk in self.decoder_blocks:
            if self.use_checkpoint:
                x, bbox_pr = checkpoint(blk, x, rois, bbox_pr_xy, bbox_pr_wh,
                                        bbox_pr_a)
            else:
                x, bbox_pr = blk(x, rois, bbox_pr_xy, bbox_pr_wh, bbox_pr_a)

            if 'XY' in blk.dc_mode_str:
                bbox_pr_xy = bbox_pr[:, 0:2]
            if 'WH' in blk.dc_mode_str:
                bbox_pr_wh = bbox_pr[:, 2:4]
            if 'A' in blk.dc_mode_str:
                bbox_pr_a = bbox_pr[:, 4:5]

        x = self.decoder_box_norm(x.mean(dim=1))
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = torch.cat([bbox_pr_xy, bbox_pr_wh, bbox_pr_a], dim=1)
        return cls_score, bbox_pred
