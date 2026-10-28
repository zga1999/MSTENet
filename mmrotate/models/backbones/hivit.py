# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, Mlp, trunc_normal_
from timm.layers import to_2tuple

from mmengine.logging import MMLogger

from mmrotate.registry import MODELS
from mmrotate.utils.mae_checkpoint import load_checkpoint_hivit_1 as load_checkpoint


# ============================================================================
# HiViT标准实现组件
# ============================================================================

class Attention(nn.Module):
    """带相对位置编码的多头注意力机制"""
    def __init__(self, input_size, dim, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., rpe=True):
        super().__init__()
        self.input_size = input_size
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * input_size - 1) * (2 * input_size - 1), num_heads)
        ) if rpe else None

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpe_index=None, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if rpe_index is not None:
            S = int(math.sqrt(rpe_index.size(-1)))
            relative_position_bias = self.relative_position_bias_table[rpe_index].view(-1, S, S, self.num_heads)
            relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous()
            assert N == S
            attn = attn + relative_position_bias
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.float().clamp(min=torch.finfo(torch.float32).min, max=torch.finfo(torch.float32).max)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BlockWithRPE(nn.Module):
    """带相对位置编码的Transformer块"""
    def __init__(self, input_size, dim, num_heads=0., mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., rpe=True,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        with_attn = num_heads > 0.

        self.norm1 = norm_layer(dim) if with_attn else None
        self.attn = Attention(
            input_size, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, rpe=rpe,
        ) if with_attn else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, rpe_index=None, mask=None):
        if self.attn is not None:
            x = x + self.drop_path(self.attn(self.norm1(x), rpe_index, mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """图像到patch嵌入"""
    def __init__(self, img_size=224, patch_size=16, inner_patches=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.inner_patches = inner_patches
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        conv_size = [size // inner_patches for size in patch_size]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=conv_size, stride=conv_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        patches_resolution = (H // self.patch_size[0], W // self.patch_size[1])
        num_patches = patches_resolution[0] * patches_resolution[1]
        x = self.proj(x).view(
            B, -1,
            patches_resolution[0], self.inner_patches,
            patches_resolution[1], self.inner_patches,
        ).permute(0, 2, 4, 3, 5, 1).reshape(B, num_patches, self.inner_patches, self.inner_patches, -1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerge(nn.Module):
    """Patch合并层,用于空间分辨率下采样"""
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x):
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class HiViT_std(nn.Module):
    """HiViT标准实现"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=512, depths=[4, 4, 20], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, ape=True, rpe=True, patch_norm=True, use_checkpoint=False,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.ape = ape
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.num_main_blocks = depths[-1]

        embed_dim = embed_dim // 2 ** (self.num_layers - 1)
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        Hp, Wp = self.patch_embed.patches_resolution
        assert Hp == Wp

        # absolute position embedding
        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.num_features)
            )
            trunc_normal_(self.absolute_pos_embed, std=.02)
        if rpe:
            coords_h = torch.arange(Hp)
            coords_w = torch.arange(Wp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += Hp - 1
            relative_coords[:, :, 1] += Wp - 1
            relative_coords[:, :, 0] *= 2 * Wp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) + sum(depths[:-1])))  # stochastic depth decay rule

        # build blocks
        self.blocks = nn.ModuleList()
        for stage_depth in depths:
            is_main_stage = embed_dim == self.num_features
            nhead = num_heads if is_main_stage else 0
            ratio = mlp_ratio if is_main_stage else stem_mlp_ratio
            # every block not in main stage include two mlp blocks
            stage_depth = stage_depth if is_main_stage else stage_depth * 2
            for i in range(stage_depth):
                self.blocks.append(
                    BlockWithRPE(
                        Hp, embed_dim, nhead, ratio, qkv_bias, qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr),
                        rpe=rpe, norm_layer=norm_layer,
                    )
                )
            if not is_main_stage:
                self.blocks.append(
                    PatchMerge(embed_dim, norm_layer)
                )
                embed_dim *= 2

        self.fc_norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, ids_keep=None, mask=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        if ids_keep is not None:
            x = torch.gather(
                x, dim=1, index=ids_keep[:, :, None, None, None].expand(-1, -1, *x.shape[2:])
            )

        for blk in self.blocks[:-self.num_main_blocks]:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        x = x[..., 0, 0, :]
        if self.ape:
            pos_embed = self.absolute_pos_embed
            if ids_keep is not None:
                pos_embed = torch.gather(
                    pos_embed.expand(B, -1, -1),
                    dim=1,
                    index=ids_keep[:, :, None].expand(-1, -1, pos_embed.shape[2]),
                )
            x += pos_embed
        x = self.pos_drop(x)

        rpe_index = None
        if self.rpe:
            if ids_keep is not None:
                B, L = ids_keep.shape
                rpe_index = self.relative_position_index
                rpe_index = torch.gather(
                    rpe_index[ids_keep, :], dim=-1, index=ids_keep[:, None, :].expand(-1, L, -1)
                ).reshape(B, -1)
            else:
                rpe_index = self.relative_position_index.view(-1)

        for blk in self.blocks[-self.num_main_blocks:]:
            x = checkpoint.checkpoint(blk, x, rpe_index, mask) if self.use_checkpoint else blk(x, rpe_index, mask)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean(dim=1)
        x = self.fc_norm(x)
        x = self.head(x)
        return x


def hivit_base(**kwargs):
    """创建HiViT base模型"""
    model = HiViT_std(
        embed_dim=512, depths=[4, 4, 20], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4.,
        rpe=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# ============================================================================
# 窗口注意力机制和MMDetection适配组件
# ============================================================================

def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class LayerNorm2d(nn.LayerNorm):
    """2D LayerNorm，专门为2D特征图设计"""
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


# ============================================================================
# HiViT的MMDetection适配版本
# ============================================================================

@MODELS.register_module()
class HiViT(HiViT_std):
    """HiViT的MMDetection适配版本，支持目标检测任务"""
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=512,
                 depths=[4, 4, 20],
                 num_heads=8,
                 stem_mlp_ratio=3.,
                 mlp_ratio=4.,
                 drop_path_rate=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 ape=True, rpe=True,
                 patch_norm=True,
                 with_fpn=False,
                 with_simple_fpn=False,
                 out_indices=[4, 9, 14, 19],
                 use_checkpoint=False,
                 global_indices=[4, 9, 14, 19],
                 window_size=None,
                 init_cfg=None,
                 last_feat=False,
                 **kwargs):
        super(HiViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            stem_mlp_ratio=stem_mlp_ratio,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape, rpe=rpe,
            patch_norm=patch_norm,
            **kwargs)

        assert not (with_fpn or with_simple_fpn) or (patch_size in (16,))
        assert not (with_fpn and with_simple_fpn)
        self.init_cfg = init_cfg
        self.patch_size = patch_size
        self.with_fpn = with_fpn
        self.with_simple_fpn = with_simple_fpn
        self.merge_indices = (depths[0] * 2 - 1, depths[0] * 2 + depths[1] * 2)
        self.out_indices = out_indices
        self.use_checkpoint = use_checkpoint
        self.global_indices = global_indices
        self.window_size = window_size

        self.last_feat = last_feat

        del self.fc_norm, self.head, self.num_classes
        if with_fpn:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                # nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            ) if 'H' not in out_indices else LayerNorm2d(embed_dim // 4, eps=1e-6)
            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            ) if 'M' not in out_indices else LayerNorm2d(embed_dim // 2, eps=1e-6)
            self.fpn3 = LayerNorm2d(embed_dim, eps=1e-6)
            self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                LayerNorm2d(embed_dim, eps=1e-6),
            )
        else:
            logger = MMLogger.get_current_instance()
            logger.info('Build model without FPN.')

    def init_weights(self):
        """初始化权重"""
        if self.init_cfg is None:
            raise ValueError
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            self.apply(self._init_weights)
            pretrained = self.init_cfg['checkpoint']
            logger = MMLogger.get_current_instance()
            if os.path.isfile(pretrained):
                load_checkpoint(self, pretrained, strict=False, logger=logger)
            else:
                raise ValueError(f"checkpoint path {pretrained} is invalid")

    def interpolate_pos_encoding(self, x, h, w):
        """位置编码插值，支持不同尺寸的输入"""
        npatch = x.shape[1]
        N = self.absolute_pos_embed.shape[1]
        if npatch == N and w == h:
            return self.absolute_pos_embed
        patch_pos_embed = self.absolute_pos_embed
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def prepare_tokens(self, x, mask=None):
        """准备输入tokens，包括多层特征提取"""
        B, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        features = []

        x = self.patch_embed(x)
        for i, blk in enumerate(self.blocks[:-self.num_main_blocks]):
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
            if i == self.merge_indices[0] and 'H' in self.out_indices:
                xp = x.reshape(B, Hp, Wp, 4, 4, -1).permute(
                    0, 5, 1, 3, 2, 4
                ).reshape(B, -1, Hp*4, Wp*4).contiguous()
                for _ in range(self.out_indices.count('H')):
                    features.append(xp)
            if i == self.merge_indices[1] and 'M' in self.out_indices:
                xp = x.reshape(B, Hp, Wp, 2, 2, -1).permute(
                    0, 5, 1, 3, 2, 4
                ).reshape(B, -1, Hp*2, Wp*2).contiguous()
                for _ in range(self.out_indices.count('M')):
                    features.append(xp)
        x = x[..., 0, 0, :]
        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, H, W)
        return self.pos_drop(x), features

    def forward(self, x):
        """前向传播，支持窗口注意力和多层特征输出"""
        B, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        x, features = self.prepare_tokens(x)
        rpe_index = self.relative_position_index.view(-1) if self.rpe else None

        for i, blk in enumerate(self.blocks[-self.num_main_blocks:]):
            window_attn = self.window_size is not None and i not in self.global_indices
            Ws = self.window_size
            if window_attn:
                x, pad_hw = window_partition(x.reshape(B, Hp, Wp, -1), Ws)
                x = x.flatten(1, 2)
            x = checkpoint.checkpoint(blk, x, rpe_index) if self.use_checkpoint else blk(x, rpe_index)
            if window_attn:
                x = window_unpartition(x.reshape(x.size(0), Ws, Ws, -1), Ws, pad_hw, (Hp, Wp))
                x = x.flatten(1, 2)
            if i in self.out_indices:
                xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
                for _ in range(self.out_indices.count(i)):
                    features.append(xp)

        if self.with_fpn:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(features)):
                features[i] = ops[i](features[i])

        if self.last_feat:
            return tuple(features), x
        return tuple(features)
