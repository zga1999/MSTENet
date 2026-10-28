# Copyright (c) OpenMMLab. All rights reserved.
from .convfc_rbbox_head import RotatedShared2FCBBoxHead
from .gv_bbox_head import GVBBoxHead
from .rotated_mae_bbox_head import RotatedMAEBBoxHead
from .rotated_mae_bbox_head_stdc import RotatedMAEBBoxHeadSTDC
from .strip_head import StripHead, StripHeadBase

__all__ = [
    'RotatedShared2FCBBoxHead', 'GVBBoxHead', 'StripHead', 'StripHeadBase',
    'RotatedMAEBBoxHead', 'RotatedMAEBBoxHeadSTDC'
]
