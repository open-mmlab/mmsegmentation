# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

from torch import Tensor, nn

from mmseg.core import add_prefix
from mmseg.core.utils import ConfigType, OptSampleList, SampleList
from mmseg.core.utils.typing import OptConfigType, OptMultiConfig
from mmseg.registry import MODELS
from .encoder_decoder import EncoderDecoder


@MODELS.register_module()
class CascadeEncoderDecoder(EncoderDecoder):
    """Cascade Encoder Decoder segmentors.

    CascadeEncoderDecoder almost the same as EncoderDecoder, while decoders of
    CascadeEncoderDecoder are cascaded. The output of previous decoder_head
    will be the input of next decoder_head.

    Args:

        num_stages (int): How many stages will be cascaded.
        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
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
    """

    def __init__(self,
                 num_stages: int,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        self.num_stages = num_stages
        super(CascadeEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        assert isinstance(decode_head, list)
        assert len(decode_head) == self.num_stages
        self.decode_head = nn.ModuleList()
        for i in range(self.num_stages):
            self.decode_head.append(MODELS.build(decode_head[i]))
        self.align_corners = self.decode_head[-1].align_corners
        self.num_classes = self.decode_head[-1].num_classes

    def encode_decode(self, batch_inputs: Tensor,
                      batch_img_metas: List[dict]) -> List[Tensor]:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(batch_inputs)
        out = self.decode_head[0].forward(x)
        for i in range(1, self.num_stages - 1):
            out = self.decode_head[i].forward(x, out)
        seg_logits_list = self.decode_head[-1].predict(x, out, batch_img_metas,
                                                       self.test_cfg)

        return seg_logits_list

    def _decode_head_forward_train(self, batch_inputs: Tensor,
                                   batch_data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        loss_decode = self.decode_head[0].loss(batch_inputs,
                                               batch_data_samples,
                                               self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode_0'))
        # get batch_img_metas
        batch_size = len(batch_data_samples)
        batch_img_metas = []
        for batch_index in range(batch_size):
            metainfo = batch_data_samples[batch_index].metainfo
            batch_img_metas.append(metainfo)

        for i in range(1, self.num_stages):
            # forward test again, maybe unnecessary for most methods.
            if i == 1:
                prev_outputs = self.decode_head[0].forward(batch_inputs)
            else:
                prev_outputs = self.decode_head[i - 1].forward(
                    batch_inputs, prev_outputs)
            loss_decode = self.decode_head[i].loss(batch_inputs, prev_outputs,
                                                   batch_data_samples,
                                                   self.train_cfg)
            losses.update(add_prefix(loss_decode, f'decode_{i}'))

        return losses

    def _forward(self,
                 batch_inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` and `gt_semantic_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(batch_inputs)

        out = self.decode_head[0].forward(x)
        for i in range(1, self.num_stages):
            # TODO support PointRend tensor mode
            out = self.decode_head[i].forward(x, out)

        return out
