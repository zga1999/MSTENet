# Copyright (c) OpenMMLab. All rights reserved.
"""Epoch-wise alpha schedule for the BBN dual-head RoI head.

The 0.x source file of this hook was not available, so it is reconstructed
from the only two things that pin it down: the ``type='BBNAlphaHook',
T_max=..., min_alpha=...`` entry in
``projects/vit/dota_vit_fpn_orientedstdcbbn_dualfocal_1x.py``, and the alpha
values printed in the 0.x training logs, which follow
``alpha = 1 - ((epoch + 1) / T_max) ** 2`` clamped from below by
``min_alpha``.

Note that
:class:`~mmrotate.models.roi_heads.BBNDualHeadFocalOrientedStandardRoIHeadimTED`
mixes its two classification losses with the fixed ``delta_c`` / ``delta_f``
weights and does not read ``bbn_alpha``, so this schedule currently only
records the value on the RoI head.
"""
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmrotate.registry import HOOKS


@HOOKS.register_module()
class BBNAlphaHook(Hook):
    """Decay ``roi_head.bbn_alpha`` once per training epoch.

    Args:
        T_max (int): Number of epochs the schedule spans; normally equal to
            ``train_cfg.max_epochs``.
        min_alpha (float): Lower bound of alpha. Defaults to 0.0.
    """

    priority = 'NORMAL'

    def __init__(self, T_max: int, min_alpha: float = 0.0):
        assert T_max > 0, f'T_max must be positive, but got {T_max}'
        self.T_max = T_max
        self.min_alpha = min_alpha

    def before_train_epoch(self, runner) -> None:
        """Set the alpha of the current epoch on the RoI head."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        roi_head = getattr(model, 'roi_head', None)
        if roi_head is None or not hasattr(roi_head, 'bbn_alpha'):
            return

        progress = (runner.epoch + 1) / self.T_max
        alpha = max(self.min_alpha, 1.0 - progress**2)
        roi_head.bbn_alpha = alpha
        runner.logger.info(f'BBNAlphaHook: epoch {runner.epoch}, '
                           f'bbn_alpha = {alpha:.4f}')
