# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import ConfigType, SampleList
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

    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        losses = dict()
        inputs_sup, data_samples_sup = multi_batch_inputs[
            'sup'], multi_batch_data_samples['sup']
        inputs_unsup, data_samples_unsup = multi_batch_inputs[
            'unsup'], multi_batch_data_samples['unsup']

        B = inputs_sup.size(0)
        inputs = torch.cat([inputs_sup, inputs_unsup], dim=0)
        data_samples = data_samples_sup + data_samples_unsup

        s_pred = self.student(inputs, data_samples, mode='tensor')
        t_pred = self.teacher(inputs, data_samples, mode='tensor')
        unsup_loss = self.loss_by_pseudo(s_pred, t_pred)
        losses.update(**unsup_loss)

        s_sup_pred = s_pred[:B]
        sup_loss = self.student.decode_head.loss_by_feat(
            s_sup_pred, data_samples_sup)
        losses.update(sup_loss)

        return losses

    # def get_pseudo_labels(self, inputs: Tensor,
    #                       data_samples: SampleList) -> SampleList:
    #     # add noise
    #     noise = torch.clamp(
    #         torch.randn_like(inputs) * self.noise_factor, -self.noise_range,
    #         self.noise_range)
    #     inputs = inputs + noise
    #     return self.teacher.predict(inputs, data_samples)

    def loss_by_pseudo(self, s_pred: Tensor, t_pred: Tensor) -> dict:
        unsup_weight = self.semi_train_cfg.get('unsup_weight', 1.)
        losses = self.consistency_loss(
            s_pred.softmax(dim=1), t_pred.softmax(dim=1)) * unsup_weight
        return dict(loss_consistency=losses)

    # def loss_by_pseudo(self, inputs: Tensor,data_samples:SampleList) -> dict:
    #     input_data_samples = copy.deepcopy(data_samples)
    #     data_samples_pred: SampleList = self.student.predict(
    #         inputs, input_data_samples)
    #     stu_preds = []
    #     teacher_preds = []
    #     for data_sample_stu, data_sample_teacher in zip(
    #             data_samples_pred, data_samples):
    #         stu_preds.append(data_sample_stu.seg_logits.data)
    #         teacher_preds.append(data_sample_teacher.seg_logits.data)
    #     stu_preds = torch.stack(stu_preds, dim=0).softmax(dim=1)
    #     teacher_preds = torch.stack(teacher_preds, dim=0).softmax(dim=1)
    #     # TODO: current_consistency_weight
    #     # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    #     losses_weight = self.semi_train_cfg.get('unsup_weight', 1.)
    #     losses = self.consistency_loss(stu_preds,
    #                                    teacher_preds) * losses_weight
    #     return dict(loss_consistency=losses)
