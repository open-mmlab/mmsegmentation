# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch.nn as nn
from mmengine.device import get_device
from mmengine.model import BaseTTAModel

from mmseg.registry import MODELS
from mmseg.utils import SampleList


@MODELS.register_module()
class EncoderDecoderTTA(BaseTTAModel):

    def __init__(self, module: Union[dict, nn.Module]):
        super().__init__(module)
        self.module.to(get_device())

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
                predictions.append(data_sample)

        return predictions
