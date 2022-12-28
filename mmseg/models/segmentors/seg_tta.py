# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from mmengine.model import BaseTTAModel
from mmengine.structures import PixelData

from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import SampleList


@MODELS.register_module()
class SegTTAModel(BaseTTAModel):

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
            seg_logits = data_samples[0].seg_logits.data
            seg_scores = torch.zeros(seg_logits.shape).to(seg_logits)
            for data_sample in data_samples:
                # check flip
                flip = data_sample.metainfo.get('flip', None)
                seg_logit = data_sample.seg_logits.data
                if flip:
                    flip_direction = data_sample.metainfo['flip_direction']
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        seg_logit = seg_logit.flip(dims=(2, ))
                    else:
                        seg_logit = seg_logit.flip(dims=(1, ))
                if self.module.out_channels > 1:
                    seg_scores += seg_logit.softmax(dim=0)
                else:
                    seg_scores += seg_logit.sigmoid()
            seg_scores /= len(data_samples)
            if self.module.out_channels == 1:
                seg_pred = (seg_scores > self.module.decode_head.threshold
                            ).to(seg_scores).squeeze(1)
            else:
                seg_pred = seg_scores.argmax(dim=0)
            data_sample = SegDataSample(
                **{
                    'pred_sem_seg': PixelData(data=seg_pred),
                    'gt_sem_seg': data_samples[0].gt_sem_seg
                })
            predictions.append(data_sample)
        return predictions
