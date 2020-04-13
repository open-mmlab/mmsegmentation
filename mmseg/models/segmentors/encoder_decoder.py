import torch
import torch.nn.functional as F

from .. import builder
from ..registry import SEGMENTORS
from .base import BaseSegmentor


@SEGMENTORS.register_module
class EncoderDecoder(BaseSegmentor):

    def __init__(self,
                 backbone,
                 decode_head,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EncoderDecoder, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.decode_head = builder.build_head(decode_head)
        if auxiliary_head is not None:
            self.auxiliary_head = builder.build_head(auxiliary_head)

        # TODO: implement sampler

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(EncoderDecoder, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            self.auxiliary_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)

        seg_logit = self.decode_head(x)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
        x = self.extract_feat(img)

        losses = dict()

        seg_logit = self.decode_head(x)
        loss_decode = self.decode_head.losses(seg_logit, gt_semantic_seg)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            auxiliary_seg_logit = self.auxiliary_head(x)
            loss_aux = self.auxiliary_head.losses(
                auxiliary_seg_logit, gt_semantic_seg, suffix='aux')
            losses.update(loss_aux)

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.decode_head.num_classes
        h_grids = (h_img - h_crop + h_stride - 1) // h_stride + 1
        w_grids = (w_img - w_crop + w_stride - 1) // w_stride + 1
        # TODO should not padding zero
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                pad_img = crop_img.new_zeros(
                    (crop_img.size(0), crop_img.size(1), h_crop, w_crop))
                pad_img[:, :, :y2 - y1, :x2 - x1] = crop_img
                pad_x = self.extract_feat(pad_img)
                pad_seg_logit = self.decode_head(pad_x)
                pad_seg_logit = F.interpolate(
                    input=pad_seg_logit,
                    size=pad_img.shape[2:],
                    mode='bilinear',
                    align_corners=True)
                # TODO DANET use exp here
                preds[:, :, y1:y2,
                      x1:x2] += pad_seg_logit[:, :, :y2 - y1, :x2 - x1]
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        if rescale:
            preds = F.interpolate(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=True)

        return preds

    def whole_inference(self, img, img_meta, rescale):
        # TODO scale
        x = self.extract_feat(img)
        seg_logit = self.decode_head(x)
        if rescale:
            seg_logit = F.interpolate(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=True)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            output = output.flip(dims=(3, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        seg_logit_list = []
        for i in range(len(imgs)):
            seg_logit_list.append(
                self.inference(imgs[i], img_metas[i], rescale))
        seg_logit = torch.stack(seg_logit_list).mean(dim=0)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
