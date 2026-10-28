import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrotate.registry import MODELS
from .simple_fpn import SimpleFPN


class MDFCBlock(nn.Module):
    """Frequency-Decoupled Multi-Domain Feature Coordination block.

    适配 STD: 输入 / 输出均为 [B, C, H, W]
    """

    def __init__(
        self,
        channels: int,
        split_ratio: float = 0.5,   # 通道拆分比例: 空间 / 频率
        reduction: int = 4,         # 通道注意力压缩率
        use_fft_global: bool = True # 是否做二次 FFT 全局分析
    ):
        super().__init__()
        assert 0.0 < split_ratio < 1.0
        self.channels = channels
        self.c_sp = int(channels * split_ratio)
        self.c_fp = channels - self.c_sp
        self.use_fft_global = use_fft_global

        # ----------------- 空间分支 (Spatial Branch) -----------------
        # 原文是 stride=2 下采样 + stride=1 卷积，这里下采样后再上采样回原尺寸
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(self.c_sp, self.c_sp, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.c_sp),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.c_sp, self.c_sp, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.c_sp),
            nn.ReLU(inplace=True),
        )

        # ----------------- 频率分支 (Frequency Branch) -----------------
        self.freq_reduce = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.c_fp, self.c_fp, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c_fp),
            nn.ReLU(inplace=True),
        )
        # 在频域做调制时用的幅度注意力
        self.freq_amp_conv = nn.Conv2d(self.c_fp, self.c_fp, kernel_size=1, bias=True)

        # ----------------- 通道注意力 (Channel Attention) -----------------
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # ----------------- Sobel 边缘先验 (Structural Prior) -----------------
        sobel_kernel_x = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]]
        ).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor(
            [[-1., -2., -1.],
             [ 0.,  0.,  0.],
             [ 1.,  2.,  1.]]
        ).view(1, 1, 3, 3)
        self.register_buffer('sobel_kernel_x', sobel_kernel_x)
        self.register_buffer('sobel_kernel_y', sobel_kernel_y)

        # 将边缘 map 投到 C 维通道上
        self.edge_proj = nn.Conv2d(1, channels, kernel_size=1, bias=False)

        # ----------------- 输出融合 & 残差 -----------------
        # concat 后是 [C_sp(空间调制) + C_sp(原始空间) + C_fp(原始频率)] = C + C_sp
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels + self.c_sp, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def _fft_gate(self, x_fp):
        """频率分支: 在频域上做幅度调制.

        Args:
            x_fp (Tensor): [B, C_fp, H/2, W/2]
        """
        B, C, H, W = x_fp.shape
        # FFT
        x_fft = torch.fft.rfft2(x_fp, norm='ortho')  # [B, C, H/2, W_fft]
        amp = torch.abs(x_fft)                       # 实数幅度 [B, C, H/2, W_fft]

        # 用 1x1 conv 在幅度域做 attention，再 sigmoid
        # 为了使用 Conv2d，这里把 (H, W_fft) 当成普通二维平面
        amp_att = self.freq_amp_conv(amp)
        amp_att = torch.sigmoid(amp_att)             # [B, C, H/2, W_fft]

        # 将实数 attention 乘到复数频谱上
        x_fft_mod = x_fft * amp_att

        # IFFT 回到空间
        x_rec = torch.fft.irfft2(x_fft_mod, s=(H, W), norm='ortho')  # [B, C, H/2, W/2]
        return x_rec

    def _edge_prior(self, x):
        """Sobel 边缘先验.

        Args:
            x (Tensor): [B, C, H, W]
        """
        gray = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        edge_x = F.conv2d(gray, self.sobel_kernel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_kernel_y, padding=1)
        edge_mag = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)   # [B, 1, H, W]
        edge_feat = self.edge_proj(edge_mag)                      # [B, C, H, W]
        return edge_feat

    def forward(self, x):
        """前向计算.

        Args:
            x (Tensor): [B, C, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.channels

        # ----------------- 通道拆分 -----------------
        x_sp, x_fp = torch.split(x, [self.c_sp, self.c_fp], dim=1)   # [B, C_sp, H, W], [B, C_fp, H, W]

        # ----------------- 空间分支 -----------------
        x_sp_down = self.spatial_branch(x_sp)                        # [B, C_sp, H/2, W/2]
        x_sp_up = F.interpolate(x_sp_down, size=(H, W),
                                mode='bilinear', align_corners=False)  # [B, C_sp, H, W]

        # ----------------- 频率分支 -----------------
        x_fp_red = self.freq_reduce(x_fp)                            # [B, C_fp, H/2, W/2]
        x_fp_rec = self._fft_gate(x_fp_red)                          # [B, C_fp, H/2, W/2]
        x_fp_up = F.interpolate(x_fp_rec, size=(H, W),
                                mode='bilinear', align_corners=False) # [B, C_fp, H, W]

        # ----------------- 空间-频率调制 -----------------
        # 这里只用频率增强后的部分对空间分支做 gating
        # 为了维度匹配，只拿前 C_sp 通道
        x_fp_gate = x_fp_up[:, :self.c_sp, :, :]                     # [B, C_sp, H, W]
        x_sf = x_sp_up * x_fp_gate                                   # [B, C_sp, H, W]

        # 与原始空间分支 concat
        x_cat = torch.cat([x_sf, x_sp], dim=1)                       # [B, 2*C_sp, H, W]
        # 再拼上原始频率分支，得到 C + C_sp 维
        x_cat_all = torch.cat([x_cat, x_fp], dim=1)                  # [B, C + C_sp, H, W]

        # ----------------- 多域协调: FFT + CA + Sobel -----------------
        # 1) 可选的二次 FFT 全局分析（这里我们只用于梯度流 & 轻微调制）
        if self.use_fft_global:
            # 对 concat 后的特征做 FFT + IFFT，构造一个 frequency-aware 残差
            x_fft = torch.fft.rfft2(x_cat_all, norm='ortho')
            amp = torch.abs(x_fft)
            amp = amp / (amp.mean(dim=(-2, -1), keepdim=True) + 1e-6)
            x_fft_mod = x_fft * amp
            x_freq_global = torch.fft.irfft2(
                x_fft_mod,
                s=(H, W),
                norm='ortho'
            )
        else:
            x_freq_global = 0.0

        # 2) 通道注意力（基于原始 x）
        ca = self.ca(x)                                              # [B, C, 1, 1]

        # 3) Sobel 边缘先验
        edge_feat = self._edge_prior(x)                              # [B, C, H, W]

        # 我们把 CA 和 edge 作用在原始 x 上，然后与 x_cat_all + x_freq_global 融合
        x_ca = x * ca                                                # [B, C, H, W]
        x_struct = x_ca + edge_feat                                  # [B, C, H, W]

        # 把结构增强的特征也 concat 进来
        x_fuse = x_cat_all + x_freq_global                           # [B, C + C_sp, H, W]
        # 沿着前 C 通道加入结构增强信息（简单相加）
        x_fuse[:, :C, :, :] = x_fuse[:, :C, :, :] + x_struct

        # ----------------- 输出 + 残差 -----------------
        out = self.out_conv(x_fuse)                                  # [B, C, H, W]
        out = out + x                                                # 残差
        return out


@MODELS.register_module()
class SimpleFPNMDFC(SimpleFPN):
    """
    在 SimpleFPN 的基础上，对部分尺度特征加入 MDFC 频率解耦多域协调模块
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 use_residual=True,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                 # ------ 新增参数 ------
                 with_mdfc=True,
                 mdfc_stages=(0, ),
                 mdfc_cfg=dict(split_ratio=0.5, reduction=4, use_fft_global=True),
                 ):
        super(SimpleFPNMDFC, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            start_level=start_level,
            end_level=end_level,
            add_extra_convs=add_extra_convs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            use_residual=use_residual,
            upsample_cfg=upsample_cfg,
            init_cfg=init_cfg,
        )

        self.with_mdfc = with_mdfc
        self.mdfc_stages = list(mdfc_stages)

        if self.with_mdfc:
            # 为每个输出层准备一个模块（不在 mdfc_stages 的位置就用 Identity）
            self.mdfc_blocks = nn.ModuleList()
            for i in range(self.num_outs):
                if i in self.mdfc_stages:
                    self.mdfc_blocks.append(
                        MDFCBlock(
                            channels=out_channels,
                            split_ratio=mdfc_cfg.get('split_ratio', 0.5),
                            reduction=mdfc_cfg.get('reduction', 4),
                            use_fft_global=mdfc_cfg.get('use_fft_global', True),
                        )
                    )
                else:
                    self.mdfc_blocks.append(nn.Identity())
        else:
            self.mdfc_blocks = None

    def forward(self, inputs):
        # 先执行原始 SimpleFPN 的前向
        outs = super(SimpleFPNMDFC, self).forward(inputs)

        if not self.with_mdfc:
            return outs

        outs = list(outs)
        # 对指定的层做 MDFC 增强
        for i in range(len(outs)):
            outs[i] = self.mdfc_blocks[i](outs[i])
        return tuple(outs)
