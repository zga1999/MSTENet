# Copyright (c) OpenMMLab. All rights reserved.
from mmrotate.registry import DATASETS
from .dota import DOTADataset


@DATASETS.register_module()
class COWDataset(DOTADataset):
    """COW dataset for rotated detection.

    Reuses the DOTA annotation layout (one ``.txt`` per image holding
    quadrilateral coordinates) with a single ``cow`` category and ``.jpg``
    images.

    Args:
        img_suffix (str): The suffix of images. Defaults to 'jpg'.
    """

    METAINFO = {'classes': ('cow', ), 'palette': [(220, 20, 60)]}

    def __init__(self, img_suffix: str = 'jpg', **kwargs) -> None:
        super().__init__(img_suffix=img_suffix, **kwargs)
