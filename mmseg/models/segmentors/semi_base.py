# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Union

import torch.nn as nn
from mmengine.model import BaseModel
from torch import Tensor

from mmseg.registry import MODELS
# from mmseg.models import BaseSegmentor
from mmseg.utils import (ConfigType, ForwardResults, OptConfigType,
                         OptMultiConfig, OptSampleList, SampleList,
                         rename_loss_dict, reweight_loss_dict)


@MODELS.register_module()
class SemiBaseSegmentor(BaseModel):

    def __init__(self,
                 student: ConfigType,
                 teacher: ConfigType,
                 data_preprocessor: OptConfigType = None,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.student = MODELS.build(student)
        self.teacher = MODELS.build(teacher)
        self.semi_train_cfg = semi_train_cfg
        self.semi_test_cfg = semi_test_cfg
        if self.semi_train_cfg.get('freeze_teacher', True):
            self.freeze(self.teacher)

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def forward(self,
                inputs: Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        losses = dict()
        inputs_sup, data_samples_sup = multi_batch_inputs[
            'sup'], multi_batch_data_samples['sup']
        inputs_unsup, data_samples_unsup = multi_batch_inputs[
            'unsup'], multi_batch_data_samples['unsup']

        losses_sup = self.loss_by_gt(inputs_sup, data_samples_sup)
        losses.update(**losses_sup)

        data_samples_unsup = self.get_pseudo_labels(inputs_unsup,
                                                    data_samples_unsup)

        losses_unsup = self.loss_by_pseudo(inputs_unsup, data_samples_unsup)
        losses.update(**losses_unsup)

        return losses

    def loss_by_gt(self, inputs: Tensor, data_samples: SampleList) -> dict:
        losses = self.student(inputs, data_samples, mode='loss')
        losses_weight = self.semi_train_cfg.get('sup_weight', 1.)
        return rename_loss_dict('sup',
                                reweight_loss_dict(losses, losses_weight))

    def get_pseudo_labels(self, inputs: Tensor,
                          data_samples: SampleList) -> SampleList:
        data_samples_pred: SampleList = self.teacher(
            inputs, data_samples, mode='predict')
        for data_sample, data_sample_pred in zip(data_samples,
                                                 data_samples_pred):
            data_sample.gt_sem_seg = data_sample_pred.pred_sem_seg
        return data_samples

    def loss_by_pseudo(self, inputs: Tensor, data_samples: SampleList) -> dict:
        losses = self.student(inputs, data_samples, mode='loss')
        losses_weight = self.semi_train_cfg.get('unsup_weight', 1.)
        return rename_loss_dict('unsup',
                                reweight_loss_dict(losses, losses_weight))

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        if self.semi_test_cfg.get('predict_on', 'teacher') == 'teacher':
            return self.teacher(
                batch_inputs, batch_data_samples, mode='predict')
        else:
            return self.student(
                batch_inputs, batch_data_samples, mode='predict')

    def _forward(self, inputs: Tensor, data_samples: SampleList) -> SampleList:
        if self.semi_test_cfg.get('forward_on', 'teacher') == 'teacher':
            return self.teacher(inputs, data_samples, mode='tensor')
        else:
            return self.student(inputs, data_samples, mode='tensor')

    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        if self.semi_test_cfg.get('extract_feat_on', 'teacher') == 'teacher':
            return self.teacher.extract_feat(inputs)
        else:
            return self.student.extract_feat(inputs)

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Add teacher and student prefixes to model parameter names."""
        if not any([
                'student' in key or 'teacher' in key
                for key in state_dict.keys()
        ]):
            keys = list(state_dict.keys())
            state_dict.update({'teacher.' + k: state_dict[k] for k in keys})
            state_dict.update({'student.' + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
