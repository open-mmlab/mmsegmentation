from torch import nn

from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class CascadeEncoderDecoder(EncoderDecoder):
    """Cascade Encoder Decoder segmentors

    CascadeEncoderDecoder almost the same as EncoderDecoder, while decoders
    of CascadeEncoderDecoder are cascaded. The output of previous decoder_head
    will be the input of next decoder_head.
    """

    def __init__(self,
                 num_stages,
                 backbone,
                 decode_head,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        self.num_stages = num_stages
        super(CascadeEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def _init_decode_head(self, decode_head):
        assert isinstance(decode_head, list)
        assert len(decode_head) == self.num_stages
        self.decode_head = nn.ModuleList()
        for i in range(self.num_stages):
            self.decode_head.append(builder.build_head(decode_head[i]))
        self.align_corners = self.decode_head[-1].align_corners

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        for i in range(self.num_stages):
            self.decode_head[i].init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def encode_decode(self, img):
        x = self.extract_feat(img)
        out = self.decode_head[0].get_seg(x)
        for i in range(1, self.num_stages):
            out = self.decode_head[i].get_seg(x, out)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        losses = dict()

        seg_logit = self.decode_head[0](x)
        loss_decode = self.decode_head[0].losses(
            seg_logit, gt_semantic_seg, suffix='decode_0')
        losses.update(loss_decode)
        for i in range(1, self.num_stages):
            seg_logit = self.decode_head[i](x, seg_logit)
            loss_decode = self.decode_head[i].losses(
                seg_logit, gt_semantic_seg, suffix=f'decode_{i}')
            losses.update(loss_decode)

        return losses
