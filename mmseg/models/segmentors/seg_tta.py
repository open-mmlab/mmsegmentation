# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from mmengine.model import BaseTTAModel
from mmengine.structures import PixelData

from mmseg.registry import MODELS
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
            logits = torch.zeros(seg_logits.shape).to(seg_logits)
            for data_sample in data_samples:
                seg_logit = data_sample.seg_logits.data
                if self.module.out_channels > 1:
                    logits += seg_logit.softmax(dim=0)
                else:
                    logits += seg_logit.sigmoid()
            logits /= len(data_samples)
            if self.module.out_channels == 1:
                seg_pred = (logits > self.module.decode_head.threshold
                            ).to(logits).squeeze(1)
            else:
                seg_pred = logits.argmax(dim=0)
            data_sample.set_data({'pred_sem_seg': PixelData(data=seg_pred)})
            if hasattr(data_samples[0], 'gt_sem_seg'):
                data_sample.set_data(
                    {'gt_sem_seg': data_samples[0].gt_sem_seg})
            data_sample.set_metainfo({'img_path': data_samples[0].img_path})
            predictions.append(data_sample)
        return predictions
