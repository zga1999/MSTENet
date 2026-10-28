# Copyright (c) OpenMMLab. All rights reserved.
"""imTED RoI head with a multi-scale feature modulator (MFM).

Ported from the mmrotate 0.x
``mmrotate/models/roi_heads/oriented_standard_roi_head_imted.py``. In 0.x this
subclassed ``OrientedStandardRoIHead``; in 1.x the oriented / rotated RoI heads
have both been folded into :class:`~mmdet.models.roi_heads.StandardRoIHead`,
which handles rotated boxes through the box-type system, so that is the base
class here. Only ``init_bbox_head`` and ``_bbox_forward`` are customised, as in
the original.
"""
import torch
import torch.nn as nn
from mmdet.models.roi_heads import StandardRoIHead

from mmrotate.registry import MODELS


@MODELS.register_module()
class OrientedStandardRoIHeadimTED(StandardRoIHead):
    """RoIHead with multi-scale feature modulator on the input of bbox head.

    When ``bbox_roi_extractor`` is a list of two extractor configs, the head
    runs in MFM mode: the second (single-scale) extractor pools the ViT
    feature, the first (multi-scale) one pools the FPN pyramid, and the two are
    fused with a learnable per-channel ``mfm_factor``.
    """

    def init_bbox_head(self, bbox_roi_extractor, bbox_head) -> None:
        """Initialize ``bbox_head`` and the (one or two) roi extractors."""
        if isinstance(bbox_roi_extractor, (list, tuple)):
            self.ms_bbox_roi_extractor = MODELS.build(bbox_roi_extractor[0])
            self.ss_bbox_roi_extractor = MODELS.build(bbox_roi_extractor[1])
            self.mfm_factor = nn.Parameter(
                torch.zeros(bbox_head['in_channels']), requires_grad=True)
            self.mfm_fc = nn.Conv2d(
                in_channels=256,
                out_channels=bbox_head['in_channels'],
                kernel_size=1)
            self.with_mfm = True
        else:
            self.bbox_roi_extractor = MODELS.build(bbox_roi_extractor)
            self.with_mfm = False
        self.bbox_head = MODELS.build(bbox_head)

    def _bbox_forward(self, x, rois) -> dict:
        """Box head forward function used in both training and testing."""
        if self.with_mfm:
            ss_bbox_feats = self.ss_bbox_roi_extractor([x[-1]], rois)
            x = [
                self.mfm_fc(x[i])
                for i in range(self.ms_bbox_roi_extractor.num_inputs)
            ]
            # multi scale
            ms_bbox_feats = self.ms_bbox_roi_extractor(
                x[:self.ms_bbox_roi_extractor.num_inputs], rois)

            factor = self.mfm_factor.reshape(1, -1, 1,
                                             1).expand_as(ms_bbox_feats)
            bbox_feats = ss_bbox_feats + ms_bbox_feats * factor
        else:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        if self.bbox_head.rois_mode is None:
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
        else:
            # `rois` does not require grad in bbox_head
            cls_score, bbox_pred = self.bbox_head(bbox_feats,
                                                  rois.clone().detach())

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results


@MODELS.register_module()
class RotatedStandardRoIHeadimTED(OrientedStandardRoIHeadimTED):
    """Alias kept for configs written against the 0.x class name.

    In 0.x ``RotatedStandardRoIHead`` and ``OrientedStandardRoIHead`` only
    differed in how proposals were converted to RoIs, which 1.x does
    automatically, so the two imTED variants are the same class here.
    """
