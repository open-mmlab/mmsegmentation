# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import mmengine
from mmcv.transforms import MultiScaleFlipAug

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class TestTimeAug(MultiScaleFlipAug):
    """Test-time augmentation with multiple scales and flipping.

    An example configuration is as followed:

    .. code-block::

        img_scale=(2048, 1024),
        img_ratios=[0.5, 1.0],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]

    After MultiScaleFLipAug with above configuration, the results are wrapped
    into lists of the same length as followed:

    .. code-block::

        dict(
            img=[...],
            img_shape=[...],
            scale=[(1024, 512), (1024, 512), (2048, 1024), (2048, 1024)]
            flip=[False, True, False, True]
            ...
        )

    Args:
        transforms (list[dict]): Transforms to be applied to each resized
            and flipped data.
        scales (tuple | list[tuple] | None): Images scales for resizing.
        scale_factor (float or tuple[float]): Scale factors for resizing.
            Defaults to None.
        allow_flip (bool): Whether apply flip augmentation. Defaults to False.
        flip_direction (str | list[str]): Flip augmentation directions,
            options are "horizontal", "vertical" and "diagonal". If
            flip_direction is a list, multiple flip augmentations will be
            applied. It has no effect when flip == False. Defaults to
            "horizontal".
        resize_cfg (dict): Base config for resizing. Defaults to
            ``dict(type='Resize', keep_ratio=True)``.
        flip_cfg (dict): Base config for flipping. Defaults to
            ``dict(type='RandomFlip')``.
    """

    def __init__(
        self,
        scales: Optional[Union[Tuple, List[Tuple]]] = None,
        scale_factor: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if scale_factor is not None:
            scale_factor = (
                scale_factor
                if isinstance(scale_factor, list) else [scale_factor])
            assert mmengine.is_list_of(scale_factor, float)

        if scales is not None:
            if isinstance(scales, tuple) and mmengine.is_list_of(
                    scale_factor, float):
                assert len(scales) == 2
                self.scales = [(int(scales[0] * factor),
                                int(scales[1] * factor))
                               for factor in scale_factor]
            else:
                self.scales = scales if isinstance(scales, list) else [scales]
            self.scale_key = 'scale'
            assert mmengine.is_list_of(self.scales, tuple)
        else:
            # if ``scales`` and ``scale_factor`` both be ``None``
            if scale_factor is None:
                self.scales = [1.0]  # type: ignore
            elif isinstance(scale_factor, list):
                self.scales = scale_factor  # type: ignore
            else:
                self.scales = [scale_factor]  # type: ignore

            self.scale_key = 'scale_factor'

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip})'
        repr_str += f'flip_direction={self.flip_direction}'
        return repr_str
