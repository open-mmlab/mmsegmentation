# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import ConfigType, SampleList, stack_batch
from .semi_base import SemiBaseSegmentor


@MODELS.register_module()
class MeanTeacher(SemiBaseSegmentor):

    def __init__(self,
                 noise_cfg: ConfigType = dict(factor=0.1, range=0.2),
                 **kwargs):
        super().__init__(**kwargs)

        self.noise_factor = noise_cfg.get('factor', 0.1)
        self.noise_range = noise_cfg.get('range', 0.2)
        self.consistency_loss = torch.nn.MSELoss(reduction='mean')

    def get_pseudo_labels(self, inputs: Tensor,
                          data_samples: SampleList) -> SampleList:
        # add noise
        noise = torch.clamp(
            torch.randn_like(inputs) * self.noise_factor, -self.noise_range,
            self.noise_range)
        inputs = inputs + noise
        data_samples_pred: SampleList = self.teacher.predict(
            inputs, data_samples)
        for data_sample, data_sample_pred in zip(data_samples,
                                                 data_samples_pred):
            data_sample.gt_sem_seg = data_sample_pred.pred_sem_seg
        return data_samples

    def loss_by_pseudo(self, inputs: Tensor, data_samples: SampleList) -> dict:
        data_samples_pred: SampleList = self.student.predict(
            inputs, data_samples)
        stu_preds = []
        teacher_preds = []
        for data_sample_stu, data_sample_teacher in zip(
                data_samples_pred, data_samples):
            stu_preds.append(data_sample_stu.pred_sem_seg.data)
            teacher_preds.append(data_sample_teacher.gt_sem_seg.data)
        stu_preds = stack_batch(stu_preds)
        teacher_preds = stack_batch(teacher_preds)
        # TODO: current_consistency_weight
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        losses_weight = self.semi_train_cfg.get('unsup_weight', 1.)
        losses = self.consistency_loss(stu_preds,
                                       teacher_preds) * losses_weight
        return dict(loss_consistency=losses)
