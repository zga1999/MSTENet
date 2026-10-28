# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Tuple, Union

import torch
from mmdet.models.detectors.two_stage import TwoStageDetector
from mmdet.structures import SampleList
from torch import Tensor

from mmrotate.registry import MODELS


@MODELS.register_module()
class RotatedimTED(TwoStageDetector):
    """Rotated implementation of `imTED <https://arxiv.org/abs/2205.09613>`_.

    The backbone optionally returns an extra single-scale ViT feature
    (``last_feat=True``) alongside the pyramid. The RPN always runs on the
    neck pyramid, while the RoI head can additionally consume that ViT
    feature, either as an extra pyramid level (``with_mfm=True``, multi-scale
    feature modulator) or on its own (``roi_skip_fpn=True``).

    Args:
        roi_skip_fpn (bool): Feed only the ViT feature to the RoI head,
            bypassing the FPN pyramid. Defaults to False.
        with_mfm (bool): Append the ViT feature to the pyramid so the RoI head
            can modulate multi-scale features with it. Must agree with the
            RoI head's own ``with_mfm``. Defaults to False.
        proposals_dim (int): Width of the dummy proposals used by
            ``_forward``/flops counting. Defaults to 5.
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 roi_skip_fpn: bool = False,
                 with_mfm: bool = False,
                 neck=None,
                 proposals_dim: int = 5,
                 data_preprocessor=None,
                 init_cfg=None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.roi_skip_fpn = roi_skip_fpn
        self.with_mfm = with_mfm
        assert self.with_mfm == self.roi_head.with_mfm, (
            f'`with_mfm` of the detector ({self.with_mfm}) must match that '
            f'of the roi_head ({self.roi_head.with_mfm})')
        self.proposals_dim = proposals_dim

    def extract_feat(
        self, batch_inputs: Tensor
    ) -> Union[Tuple[Tensor], Tuple[Tuple, Tensor]]:
        """Extract features from the backbone and neck.

        Returns either the neck pyramid, or a ``(pyramid, vit_feat)`` pair
        when the backbone is configured with ``last_feat=True``.
        """
        x = self.backbone(batch_inputs)

        if self._has_vit_feat(x):
            x, vit_feat = x
            if self.with_neck:
                x = self.neck(x)
            return x, vit_feat

        if self.with_neck:
            x = self.neck(x)
        return x

    @staticmethod
    def _has_vit_feat(feats) -> bool:
        """Whether the backbone returned a ``(pyramid, vit_feat)`` pair.

        The ViT feature is a 3-D ``(B, N, C)`` token sequence, which is what
        distinguishes it from the 4-D maps of a plain pyramid.
        """
        return (len(feats) == 2 and isinstance(feats[1], Tensor)
                and feats[1].dim() == 3)

    def get_roi_feat(self, x, vit_feat):
        """Build the feature list consumed by the RoI head."""
        B, _, H, W = x[2].shape
        N = vit_feat.size(1)  # B, N, C
        M = N - H * W  # see whether there is a cls token or not
        assert M == 0 or M == 1  # check M
        if self.with_mfm:
            fea = [x[i] for i in range(len(x))]
            reshaped_vit = vit_feat[:, M:, :].transpose(1, 2).reshape(
                B, -1, H, W).contiguous()
            fea.append(reshaped_vit)
            x = tuple(fea)
        else:
            if self.roi_skip_fpn:
                x = [
                    vit_feat[:, M:, :].transpose(1, 2).reshape(
                        B, -1, H, W).contiguous()
                ]
        return x

    def _split_feats(self, batch_inputs: Tensor):
        """Return ``(rpn_feats, roi_feats)`` for a batch of images."""
        feats = self.extract_feat(batch_inputs)
        if self._has_vit_feat(feats):
            x, vit_feat = feats
            return x, self.get_roi_feat(x, vit_feat)
        return feats, feats

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        x, roi_feats = self._split_feats(batch_inputs)

        losses = dict()

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            for key in list(rpn_losses.keys()):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(roi_feats, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x, roi_feats = self._split_feats(batch_inputs)

        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            roi_feats, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList = None) -> tuple:
        """Network forward process, used for flops counting and debugging."""
        results = ()
        x, roi_feats = self._split_feats(batch_inputs)

        if self.with_rpn:
            rpn_outs = self.rpn_head.forward(x)
            results = results + (rpn_outs, )

        proposals = torch.randn(1000, self.proposals_dim).to(
            batch_inputs.device)
        rois = torch.cat(
            [proposals.new_zeros(proposals.size(0), 1), proposals], dim=-1)
        roi_outs = self.roi_head._bbox_forward(roi_feats, rois)
        results = results + (roi_outs, )
        return results
