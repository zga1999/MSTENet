import math
import warnings
from functools import partial
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule, constant_init, normal_init, trunc_normal_init
from torch.nn.modules.utils import _pair as to_2tuple

from mmrotate.registry import MODELS


class DWConv(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwconv(x)


class Mlp(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BasicConv(nn.Module):

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        relu: bool = True,
        bn: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def make_divisible(value, divisor, min_value=None, min_ratio=0.9):

    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value:
        new_value += divisor
    return int(new_value)


class scale_transformation(nn.Module):
    """Scale Transformation Module (STM).

    Paper figure::

        X → Conv1x1 ─┬─► Conv3x3 ─────────────────────┐
                     ├─► Conv1x5 → Conv5x1 → Conv5x5 ─┼─► ⊕ → ConvLinear → ⊕ → X̃
                     └─► Conv7x1 → Conv1x7 → Conv7x7 ─┘                       ↑
        X ──────────────────────── shortcut ──────────────────────────────────┘

    ``scale`` damps the multi-branch path before the residual add, as in the RFB
    block this module derives from.
    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        scale: float = 0.1,
        map_reduce: int = 8,
    ):
        super().__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter = max(1, in_planes // map_reduce)

        self.reduce = BasicConv(in_planes, inter, kernel_size=1, stride=1)

        self.branch0 = BasicConv(
            inter, inter, kernel_size=3, stride=stride, padding=1, relu=False)

        self.branch1 = nn.Sequential(
            BasicConv(
                inter, inter, kernel_size=(1, 5), stride=stride, padding=(0, 2)),
            BasicConv(inter, inter, kernel_size=(5, 1), stride=1, padding=(2, 0)),
            BasicConv(
                inter, inter, kernel_size=5, stride=1, padding=2, relu=False),
        )

        self.branch2 = nn.Sequential(
            BasicConv(
                inter, inter, kernel_size=(7, 1), stride=stride, padding=(3, 0)),
            BasicConv(inter, inter, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv(
                inter, inter, kernel_size=7, stride=1, padding=3, relu=False),
        )

        self.ConvLinear = BasicConv(
            inter, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(
            in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.reduce(x)
        out = self.branch0(r) + self.branch1(r) + self.branch2(r)
        out = self.ConvLinear(out)
        return self.relu(out * self.scale + self.shortcut(x))


class fine_grained(BaseModule):
    """Fine-grained Enhancement Module (FEM).

    ``AvgPool → 1x1 → DW 1x11 → DW 11x1 → 1x1 → pointwise sigmoid``
    """

    def __init__(
        self,
        channels: int,
        h_kernel_size: int = 11,
        v_kernel_size: int = 11,
        norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: Optional[dict] = dict(type='SiLU'),
        init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(
            channels, channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(
            channels, channels, (1, h_kernel_size), 1,
            (0, h_kernel_size // 2), groups=channels, norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(
            channels, channels, (v_kernel_size, 1), 1,
            (v_kernel_size // 2, 0), groups=channels, norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(
            channels, channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The sigmoid is applied back onto Y, so GFF fuses two feature maps
        # (STM features ⊕ FEM features) rather than a feature and a raw mask.
        attn = self.act(
            self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return x * attn


class MSTEblock(nn.Module):
    """MSTE block, the LSKmodule replacement drawn in the paper figure::

        U → X = DW5x5      ──► STM ──► X̃ ─┐
                                           ⊕ → Z → GFF → Z̃ ─⊗ U → Ũ
        U → Y = DW7x7 (d=3) ──► FEM ──► Ỹ ─┘

    GFF produces a local/global channel gate ``m`` and two spatial weights
    ``att1/att2`` from the avg/max squeeze of ``[X̃, Ỹ]``::

        Z̃ = att1·X̃·m + att2·Ỹ·(1 - m)

    ``conv0`` / ``conv_spatial`` / ``conv_squeeze`` keep MSTENet's names and
    shapes so an MSTENet ImageNet checkpoint still loads into those weights.
    """

    def __init__(
        self,
        channels: int = 64,
        r: int = 4,
        map_reduce: int = 8,
        global_pool_size: int = 1,
        squeeze_kernel: int = 7,
    ):
        super().__init__()
        inter_channels = max(1, channels // r)

        # U → X and U → Y, two parallel large-kernel depth-wise stems
        self.conv0 = nn.Conv2d(
            channels, channels, 5, padding=2, groups=channels)
        self.conv_spatial = nn.Conv2d(
            channels, channels, 7, stride=1, padding=9, dilation=3,
            groups=channels)

        self.st = scale_transformation(
            channels, channels, stride=1, map_reduce=map_reduce)
        self.fg = fine_grained(channels, 11, 11, None, None)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, 1),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((global_pool_size, global_pool_size)),
            nn.Conv2d(channels, inter_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, 1),
        )

        self.conv_squeeze = nn.Conv2d(
            2, 2, kernel_size=squeeze_kernel, padding=squeeze_kernel // 2)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        x_tilde = self.st(self.conv0(u))
        y_tilde = self.fg(self.conv_spatial(u))

        z = x_tilde + y_tilde

        xl = self.local_att(z)
        xg = self.global_att(z)
        if xg.shape[2:] != (1, 1) and xg.shape[2:] != xl.shape[2:]:
            xg = F.interpolate(
                xg, size=xl.shape[2:], mode='bilinear', align_corners=False)
        m = torch.sigmoid(xl + xg)

        feat = torch.cat([x_tilde, y_tilde], dim=1)
        avg_attn = torch.mean(feat, dim=1, keepdim=True)
        max_attn, _ = torch.max(feat, dim=1, keepdim=True)
        att = self.conv_squeeze(
            torch.cat([avg_attn, max_attn], dim=1)).sigmoid()

        z_tilde = (x_tilde * att[:, 0:1] * m
                   + y_tilde * att[:, 1:2] * (1.0 - m))
        return u * z_tilde


class Attention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = MSTEblock(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


class Block(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4., drop: float = 0., drop_path: float = 0.,
                 act_layer=nn.GELU, norm_cfg: Optional[dict] = None):
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)

        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):

    def __init__(self, img_size: int = 224, patch_size: int = 7, stride: int = 4,
                 in_chans: int = 3, embed_dim: int = 768, norm_cfg: Optional[dict] = None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


@MODELS.register_module()
class MSTENet(BaseModule):

    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        embed_dims: Sequence[int] = (64, 128, 256, 512),
        mlp_ratios: Sequence[float] = (8, 8, 4, 4),
        drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths: Sequence[int] = (3, 4, 6, 3),
        num_stages: int = 4,
        num_classes: int = 0,
        pretrained: Optional[str] = None,
        init_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), 'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.depths = list(depths)
        self.num_stages = num_stages

        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dims[-1])
        if self.num_classes > 0:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(self.embed_dim, self.num_classes)
        else:
            self.global_pool = None
            self.head = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                norm_cfg=norm_cfg,
            )

            block = nn.ModuleList([
                Block(
                    dim=embed_dims[i],
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    norm_cfg=norm_cfg,
                )
                for j in range(self.depths[i])
            ])

            norm = norm_layer(embed_dims[i])
            cur += self.depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=0.02, bias=0.0)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, mean=0.0, std=math.sqrt(2.0 / fan_out), bias=0.0)
        else:
            super().init_weights()

    def freeze_patch_emb(self):
        for p in getattr(self, "patch_embed1").parameters():
            p.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: str = ''):
        self.num_classes = int(num_classes)
        if self.num_classes > 0:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(self.embed_dim, self.num_classes)
        else:
            self.global_pool = None
            self.head = None

    def forward_features(self, x: torch.Tensor):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)

            x = x.flatten(2).transpose(1, 2)  # (B,HW,C)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

    def forward(self, x: torch.Tensor):
        outs = self.forward_features(x)
        if self.num_classes > 0:
            x = outs[-1]
            x = self.global_pool(x).flatten(1)  # type: ignore[union-attr]
            x = self.head(x)                    # type: ignore[misc]
            return x
        return outs


def _conv_filter(state_dict, patch_size: int = 16):
    """Optional helper for converting manual patchify weights to conv weights."""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict
