# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
from ..utils import resize

from mmseg.registry import MODELS
from .base import BaseSegmentor
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
from torch import Tensor
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
from mmseg.utils import ConfigType, OptConfigType, OptMultiConfig, OptSampleList, SampleList, add_prefix
import logging
from mmengine.logging import print_log

@MODELS.register_module()
class MultitaskEncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    Multi task EncoderDecoder typically consists of backbone, multiple decode_heads, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.
    (this is done for each decode head)

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (List): The list of the config for the decode heads of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501
    def __init__(self,
                 backbone: ConfigType,
                 decode_head: List[ConfigType],
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_heads(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # assert self.with_decode_head

    def _init_decode_heads(self, decode_heads: List[ConfigType]) -> None:
        """Initialize multiple decode heads."""
        self.decode_heads = nn.ModuleDict()
        for head_cfg in decode_heads:
            task = head_cfg.pop('task')
            self.decode_heads[task] = MODELS.build(head_cfg)
        self.align_corners = self.decode_heads[task].align_corners

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor, batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)
        seg_logits = {}
        for task, head in self.decode_heads.items():
            seg_logits[task] = head.predict(x, batch_img_metas, self.test_cfg)
        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor], data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in training."""
        losses = dict()
        for task_index, (task, head) in enumerate(self.decode_heads.items()):
            loss_decode = head.loss(inputs, data_samples, self.train_cfg, task_index)
            losses.update(add_prefix(loss_decode, task))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor], data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))
        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        x = self.extract_feat(inputs)
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)
        return losses

    def predict(self, inputs: Tensor, data_samples: OptSampleList = None) -> Dict[str, Tensor]:
        """Predict results from a batch of inputs and data samples with post-processing."""
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0]
                )
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)
        for task in self.decode_heads.keys():
            data_samples = self.postprocess_result(seg_logits[task], task, data_samples)
        
        return data_samples

    def _forward(self, inputs: Tensor, data_samples: OptSampleList = None) -> Dict[str, Tensor]:
        """Network forward process."""
        x = self.extract_feat(inputs)
        results = {}
        for task, head in self.decode_heads.items():
            results[task] = head.forward(x)
        return results

    def slide_inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        """Inference by sliding-window with overlap."""
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        seg_logits = {task: inputs.new_zeros((batch_size, head.out_channels, h_img, w_img))
                      for task, head in self.decode_heads.items()}
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                crop_seg_logits = self.encode_decode(crop_img, batch_img_metas)
                for task, crop_seg_logit in crop_seg_logits.items():
                    seg_logits[task] += F.pad(crop_seg_logit,
                                              (int(x1), int(seg_logits[task].shape[3] - x2), int(y1),
                                               int(seg_logits[task].shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        for task in seg_logits:
            seg_logits[task] /= count_mat
        return seg_logits

    def whole_inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        """Inference with full image."""
        seg_logits = self.encode_decode(inputs, batch_img_metas)
        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Dict[str, Tensor]:
        """Inference with slide/whole style."""
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got {self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logits = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logits = self.whole_inference(inputs, batch_img_metas)
        return seg_logits

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations."""
        assert rescale
        seg_logits = self.inference(inputs[0], batch_img_metas[0])
        for i in range(1, len(inputs)):
            cur_seg_logits = self.inference(inputs[i], batch_img_metas[i])
            for task in seg_logits:
                seg_logits[task] += cur_seg_logits[task]
        for task in seg_logits:
            seg_logits[task] /= len(inputs)
        seg_pred = {task: seg_logit.argmax(dim=1) for task, seg_logit in seg_logits.items()}
        seg_pred = {task: list(pred) for task, pred in seg_pred.items()}
        return seg_pred
    def postprocess_result(self,
                           seg_logits: Tensor,
                           task,
                           data_samples: OptSampleList = None) -> SampleList:
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # resize as original shape
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits >
                              self.decode_head.threshold).to(i_seg_logits)
            data_samples[i].set_data({
                ' ':
                PixelData(**{'data': i_seg_logits}),
                f'pred_sem_seg_{task}':
                PixelData(**{'data': i_seg_pred})
            })

        return data_samples
