# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.roi_heads.bbox_heads import BBoxHead
from mmengine.config import ConfigDict

from mmrotate.registry import MODELS


class StripBlock(nn.Module):
    """Strip attention block used in the regression branch of StripHead.

    A 1x19 and a 19x1 depth-wise convolution jointly cover a large receptive
    field along both axes, which suits the elongated objects common in
    remote sensing imagery.
    """

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.strip_conv1 = nn.Conv2d(
            dim, dim, kernel_size=(1, 19), stride=1, padding=(0, 9),
            groups=dim)
        self.strip_conv2 = nn.Conv2d(
            dim, dim, kernel_size=(19, 1), stride=1, padding=(9, 0),
            groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.strip_conv1(attn)
        attn = self.strip_conv2(attn)
        attn = self.conv1(attn)

        return u * attn


@MODELS.register_module()
class StripHeadBase(BBoxHead):
    r"""Bbox head of Strip R-CNN.

    Unlike a standard ConvFC head, the regression branch is split into an
    ``xywh`` branch (equipped with :class:`StripBlock`) and a separate
    ``theta`` branch, whose outputs are concatenated into the final 5-D
    rotated box prediction.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs  |-> reg xywh convs (strip) -> fcs -\
                                    \-> reg theta convs -> theta fcs --+-> reg

    Args:
        num_shared_convs (int): number of ``shared_convs``.
        num_shared_fcs (int): number of ``shared_fcs``.
        num_cls_convs (int): number of ``cls_convs``.
        num_cls_fcs (int): number of ``cls_fcs``.
        num_reg_xy_wh_convs (int): number of strip convs in the xywh branch.
        num_reg_xy_wh_fcs (int): number of fcs in the xywh branch.
        num_reg_theta_convs (int): number of convs in the theta branch.
        num_reg_theta_fcs (int): number of fcs in the theta branch.
        conv_out_channels (int): output channels of convolution.
        fc_out_channels (int): output channels of fc.
        conv_cfg (dict, optional): Config of convolution.
        norm_cfg (dict, optional): Config of normalization.
        init_cfg (dict, optional): Config of initialization.
    """

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_xy_wh_convs=0,
                 num_reg_xy_wh_fcs=0,
                 num_reg_theta_convs=0,
                 num_reg_theta_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(StripHeadBase, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_xy_wh_convs = num_reg_xy_wh_convs
        self.num_reg_xy_wh_fcs = num_reg_xy_wh_fcs

        self.num_reg_theta_convs = num_reg_theta_convs
        self.num_reg_theta_fcs = num_reg_theta_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg xywh specific branch
        self.reg_xy_wh_convs, self.reg_xy_wh_fcs, self.reg_xy_wh_last_dim = \
            self._add_conv_strip_fc_branch(
                self.num_reg_xy_wh_convs, self.num_reg_xy_wh_fcs,
                self.shared_out_channels)

        # add reg theta specific branch
        self.reg_theta_convs, self.reg_theta_fcs, self.reg_theta_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_theta_convs, self.num_reg_theta_fcs,
                self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_xy_wh_fcs == 0:
                self.reg_xy_wh_last_dim *= self.roi_feat_area
            if self.num_reg_theta_fcs == 0:
                self.reg_theta_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            cls_predictor_cfg_ = self.cls_predictor_cfg.copy()
            cls_predictor_cfg_.update(
                in_features=self.cls_last_dim, out_features=cls_channels)
            self.fc_cls = MODELS.build(cls_predictor_cfg_)
        if self.with_reg:
            # the coder encodes a rotated box as (dx, dy, dw, dh, dtheta);
            # the first four are produced by the strip branch and the angle
            # by the theta branch.
            box_dim = self.bbox_coder.encode_size
            out_dim_reg_xy_wh = ((box_dim - 1) if self.reg_class_agnostic else
                                 (box_dim - 1) * self.num_classes)
            out_dim_reg_theta = (1 if self.reg_class_agnostic else
                                 1 * self.num_classes)
            reg_xy_wh_predictor_cfg_ = self.reg_predictor_cfg.copy()
            if isinstance(reg_xy_wh_predictor_cfg_, (dict, ConfigDict)):
                reg_xy_wh_predictor_cfg_.update(
                    in_features=self.reg_xy_wh_last_dim,
                    out_features=out_dim_reg_xy_wh)
            self.fc_reg_xy_wh = MODELS.build(reg_xy_wh_predictor_cfg_)

            reg_theta_predictor_cfg_ = self.reg_predictor_cfg.copy()
            if isinstance(reg_theta_predictor_cfg_, (dict, ConfigDict)):
                reg_theta_predictor_cfg_.update(
                    in_features=self.reg_theta_last_dim,
                    out_features=out_dim_reg_theta)
            self.fc_reg_theta = MODELS.build(reg_theta_predictor_cfg_)

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_xy_wh_fcs'),
                        dict(name='reg_theta_fcs')
                    ])
            ]

    def _add_conv_strip_fc_branch(self,
                                  num_branch_convs,
                                  num_branch_fcs,
                                  in_channels,
                                  is_shared=False):
        """Add a branch whose convs are each followed by a StripBlock."""
        last_layer_dim = in_channels
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
                branch_convs.append(StripBlock(self.conv_out_channels))
            last_layer_dim = self.conv_out_channels
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        """Forward function."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        x_reg_xy_wh = x_reg
        for conv in self.reg_xy_wh_convs:
            x_reg_xy_wh = conv(x_reg_xy_wh)
        if x_reg_xy_wh.dim() > 2:
            if self.with_avg_pool:
                x_reg_xy_wh = self.avg_pool(x_reg_xy_wh)
            x_reg_xy_wh = x_reg_xy_wh.flatten(1)
        for fc in self.reg_xy_wh_fcs:
            x_reg_xy_wh = self.relu(fc(x_reg_xy_wh))

        x_reg_theta = x_reg
        for conv in self.reg_theta_convs:
            x_reg_theta = conv(x_reg_theta)
        if x_reg_theta.dim() > 2:
            if self.with_avg_pool:
                x_reg_theta = self.avg_pool(x_reg_theta)
            x_reg_theta = x_reg_theta.flatten(1)
        for fc in self.reg_theta_fcs:
            x_reg_theta = self.relu(fc(x_reg_theta))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        xy_wh_pred = self.fc_reg_xy_wh(x_reg_xy_wh) if self.with_reg else None
        theta_pred = self.fc_reg_theta(x_reg_theta) if self.with_reg else None
        bbox_pred = torch.cat((xy_wh_pred, theta_pred), dim=1)
        return cls_score, bbox_pred


@MODELS.register_module()
class StripHead(StripHeadBase):
    """StripHead with the layer counts used by Strip R-CNN."""

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(StripHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=0,
            num_cls_convs=0,
            num_cls_fcs=2,
            num_reg_xy_wh_convs=1,
            num_reg_xy_wh_fcs=0,
            num_reg_theta_convs=0,
            num_reg_theta_fcs=2,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
