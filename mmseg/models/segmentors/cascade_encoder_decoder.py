# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.registry import MODELS
from .encoder_decoder import EncoderDecoder


@MODELS.register_module()
class CascadeEncoderDecoder(EncoderDecoder):
    """Cascade Encoder Decoder segmentors.

    CascadeEncoderDecoder almost the same as EncoderDecoder, while decoders of
    CascadeEncoderDecoder are cascaded. The output of previous decoder_head
    will be the input of next decoder_head.
    """

    def __init__(self,
                 num_stages,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 preprocess_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        self.num_stages = num_stages
        super(CascadeEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            preprocess_cfg=preprocess_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        assert isinstance(decode_head, list)
        assert len(decode_head) == self.num_stages
        self.decode_head = nn.ModuleList()
        for i in range(self.num_stages):
            self.decode_head.append(MODELS.build(decode_head[i]))
        self.align_corners = self.decode_head[-1].align_corners
        self.num_classes = self.decode_head[-1].num_classes

    def encode_decode(self, batch_inputs, batch_img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(batch_inputs)
        out = self.decode_head[0].forward_test(x, batch_img_metas,
                                               self.test_cfg)
        for i in range(1, self.num_stages):
            out = self.decode_head[i].forward_test(x, out, batch_img_metas,
                                                   self.test_cfg)
        out = resize(
            input=out,
            size=batch_inputs.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, batch_inputs, batch_data_samples):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        loss_decode = self.decode_head[0].forward_train(
            batch_inputs, batch_data_samples, self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode_0'))
        # get batch_img_metas
        batch_size = len(batch_data_samples)
        batch_img_metas = []
        for batch_index in range(batch_size):
            metainfo = batch_data_samples[batch_index].metainfo
            metainfo['batch_input_shape'] = \
                tuple(batch_inputs[batch_index].size()[-2:])
            batch_img_metas.append(metainfo)

        for i in range(1, self.num_stages):
            # forward test again, maybe unnecessary for most methods.
            if i == 1:
                prev_outputs = self.decode_head[0].forward_test(
                    batch_inputs, batch_img_metas, self.test_cfg)
            else:
                prev_outputs = self.decode_head[i - 1].forward_test(
                    batch_inputs, prev_outputs, batch_img_metas, self.test_cfg)
            loss_decode = self.decode_head[i].forward_train(
                batch_inputs, prev_outputs, batch_data_samples, self.train_cfg)
            losses.update(add_prefix(loss_decode, f'decode_{i}'))

        return losses
