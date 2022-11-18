# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from mmengine.model import BaseTTAModel

from mmseg.registry import MODELS
from mmseg.utils import SampleList


@MODELS.register_module()
class EncoderDecoderTTA(BaseTTAModel):

    def merge_preds(self, data_samples_list: List[SampleList]) -> SampleList:
        """Merge predictions of enhanced data to one prediction.

        Args:
            data_samples_list (List[SampleList]): List of predictions
                of all enhanced data.

        Returns:
            SampleList: Merged prediction.
        """
        predictions = []
        for data_samples in data_samples_list:
            for data_sample in data_samples:
                # check flip
                flip = data_sample.metainfo.get('flip', None)
                if flip:
                    pred_sem_seg = data_sample.pred_sem_seg.data
                    direction = data_sample.metainfo['flip_direction']
                    if direction == 'vertical':
                        dim = 0
                    elif direction == 'horizontal':
                        dim = 1
                    else:
                        raise f'unexpected flip direction, got {direction}'
                    pred_sem_seg = torch.flip(pred_sem_seg, dims=[dim])
                    data_sample.pred_sem_seg.data = pred_sem_seg
                predictions.append(data_sample)
        return predictions
