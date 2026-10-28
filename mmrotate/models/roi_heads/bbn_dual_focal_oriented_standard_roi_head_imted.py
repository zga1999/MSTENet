# Copyright (c) OpenMMLab. All rights reserved.
"""Dual-classifier (CE + Focal) imTED RoI head.

Ported from the mmrotate 0.x
``mmrotate/models/roi_heads/bbn_dual_focal_oriented_standard_roi_head_imted.py``.
The 0.x ``_bbox_forward_train`` hook point became ``bbox_loss`` in 1.x, and
``rbbox2roi`` / ``res.bboxes`` became ``bbox2roi`` / ``res.priors``; the loss
composition itself is unchanged.
"""
import torch
import torch.nn.functional as F
from mmdet.structures.bbox import bbox2roi

from mmrotate.registry import MODELS
from .oriented_standard_roi_head_imted import OrientedStandardRoIHeadimTED


@MODELS.register_module()
class BBNDualHeadFocalOrientedStandardRoIHeadimTED(OrientedStandardRoIHeadimTED
                                                   ):
    """Two RoI classification heads mixed in the BBN style.

    - branch C (``bbox_head_c``): plain CE, drives both cls and bbox;
    - branch F (``bbox_head_f``): Focal loss, classification only.

    Training uses ``loss_cls = delta_c * loss_cls_c + delta_f * loss_cls_f``
    and ``loss_bbox = loss_bbox_c``. Testing is inherited from the parent and
    therefore only uses branch C.

    Args:
        bbox_head_c (dict): Config of the CE branch (also becomes
            ``self.bbox_head``, i.e. the branch used at test time).
        bbox_head_f (dict): Config of the Focal branch; usually the same
            architecture as branch C.
        delta_f (float): Weight of the Focal classification loss.
            Defaults to 0.1.
        delta_c (float): Weight of the CE classification loss. Defaults to 1.
        bbn_min_alpha (float): Lower bound of ``bbn_alpha``. Defaults to 0.0.
        focal_gamma (float): Focal loss :math:`\\gamma`. Defaults to 2.0.
    """

    def __init__(self,
                 bbox_head_c,
                 bbox_head_f,
                 delta_f=0.1,
                 delta_c=1,
                 bbn_min_alpha=0.0,
                 focal_gamma=2.0,
                 **kwargs):
        self.delta_f = delta_f
        self.delta_c = delta_c

        # the parent needs a `bbox_head`; branch C plays that role
        super().__init__(bbox_head=bbox_head_c, **kwargs)

        self.bbox_head_c = self.bbox_head
        self.bbox_head_f = MODELS.build(bbox_head_f)

        self.bbn_min_alpha = bbn_min_alpha
        self.focal_gamma = focal_gamma

        # updated during training by `BBNAlphaHook`
        self.bbn_alpha = 1.0

    def _compute_focal_cls_loss(self, cls_score, labels, label_weights):
        """Multi-class softmax Focal Loss, used by branch F only.

        Args:
            cls_score (Tensor): Shape (N, num_classes + 1).
            labels (Tensor): Shape (N, ); 0~C-1 foreground, C background,
                -1 ignored.
            label_weights (Tensor): Shape (N, ).

        Returns:
            Tensor: Scalar focal loss.
        """
        valid_mask = labels >= 0
        if valid_mask.sum() == 0:
            return cls_score.new_tensor(0.0)

        labels_valid = labels[valid_mask]
        label_weights_valid = label_weights[valid_mask]
        cls_score_valid = cls_score[valid_mask]

        log_p = F.log_softmax(cls_score_valid, dim=1)
        log_p_true = log_p[torch.arange(log_p.size(0)), labels_valid]
        p_true = log_p_true.exp()

        focal_factor = (1.0 - p_true)**self.focal_gamma

        loss_per_sample = -focal_factor * log_p_true
        loss = (loss_per_sample * label_weights_valid).sum() / \
            label_weights_valid.sum().clamp(min=1.0)

        return loss

    def bbox_loss(self, x, sampling_results) -> dict:
        """Run both branches on shared RoI features and mix their losses."""
        rois = bbox2roi([res.priors for res in sampling_results])

        # branch C, `self.bbox_head` is `self.bbox_head_c`
        bbox_results = self._bbox_forward(x, rois)
        cls_score_c = bbox_results['cls_score']
        bbox_pred_c = bbox_results['bbox_pred']
        bbox_feats = bbox_results['bbox_feats']

        # branch F reuses the very same RoI features
        if getattr(self.bbox_head_f, 'rois_mode', None) is None:
            cls_score_f, _ = self.bbox_head_f(bbox_feats)
        else:
            cls_score_f, _ = self.bbox_head_f(bbox_feats,
                                              rois.clone().detach())

        labels, label_weights, bbox_targets, bbox_weights = \
            self.bbox_head_c.get_targets(
                sampling_results, self.train_cfg, concat=True)

        loss_dict_c = self.bbox_head_c.loss(cls_score_c, bbox_pred_c, rois,
                                            labels, label_weights,
                                            bbox_targets, bbox_weights)
        loss_cls_c = loss_dict_c['loss_cls']
        loss_bbox_c = loss_dict_c['loss_bbox']

        loss_cls_f = self._compute_focal_cls_loss(cls_score_f, labels,
                                                  label_weights)

        loss_cls = self.delta_c * loss_cls_c + self.delta_f * loss_cls_f

        losses = dict(loss_cls=loss_cls, loss_bbox=loss_bbox_c)
        # keep the extra statistics of branch C, e.g. `acc`
        for k, v in loss_dict_c.items():
            if k not in ['loss_cls', 'loss_bbox']:
                losses[k] = v

        bbox_results.update(loss_bbox=losses)
        return bbox_results
