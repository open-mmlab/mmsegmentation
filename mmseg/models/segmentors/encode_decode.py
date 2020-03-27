import torch
import torch.nn.functional as F

from .. import builder
from ..registry import SEGMENTORS
from .base import BaseSegmentor


@SEGMENTORS.register_module
class EncodeDecode(BaseSegmentor):

    def __init__(self,
                 backbone,
                 decode_head,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EncodeDecode, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.decode_head = builder.build_head(decode_head)
        if auxiliary_head is not None:
            self.auxiliary_head = builder.build_head(auxiliary_head)

        # TODO: implement sampler

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(EncodeDecode, self).init_weights(pretrained)
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
    def slide_inference(self, img, img_meta, rescale=False):
        # TODO scale
        stride = self.test_cfg.stride
        crop_scale = self.test_cfg.crop_scale
        batch_size, _, img_height, img_width = img.size()
        num_classes = len(self.CLASSES)
        height_grids = (img_height - crop_scale + stride - 1) // stride + 1
        width_grids = (img_width - crop_scale + stride - 1) // stride + 1
        preds = img.new_zeros((batch_size, num_classes, img_height, img_width))
        count_mat = img.new_zeros((batch_size, 1, img_height, img_width))
        for h_idx in range(height_grids):
            for w_idx in range(width_grids):
                y1 = h_idx * stride
                x1 = w_idx * stride
                y2 = min(y1 + crop_scale, img_height)
                x2 = min(x1 + crop_scale, img_width)
                y1 = max(y2 - crop_scale, 0)
                x1 = max(x2 - crop_scale, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                pad_img = crop_img.new_zeros(
                    (crop_img.size(0), crop_img.size(1), crop_scale,
                     crop_scale))
                pad_img[:, :, :y2 - y1, :x2 - x1] = crop_img
                pad_x = self.extract_feat(pad_img)
                pad_seg_logit = self.decode_head(pad_x)
                pad_seg_logit = F.interpolate(
                    input=pad_seg_logit,
                    size=(crop_scale, crop_scale),
                    mode='bilinear',
                    align_corners=True)
                # TODO DANET use exp here
                preds[:, :, y1:y2,
                      x1:x2] += pad_seg_logit[:, :, :y2 - y1, :x2 - x1]
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        ori_img_height, ori_img_width, _ = img_meta[0]['ori_shape']
        preds = F.interpolate(
            preds,
            size=(ori_img_height, ori_img_width),
            mode='bilinear',
            align_corners=True)

        return preds

    def whole_inference(self, img, img_meta, rescale):
        # TODO scale
        x = self.extract_feat(img)
        seg_logit = self.decode_head(x)
        ori_img_height, ori_img_width, _ = img_meta[0]['ori_shape']
        seg_logit = F.interpolate(
            seg_logit,
            size=(ori_img_height, ori_img_width),
            mode='bilinear',
            align_corners=True)

        return seg_logit

    def simple_test(self, img, img_meta, rescale=False):
        seg_logit = self.whole_inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        seg_logit_list = []
        for i in range(len(imgs)):
            seg_logit_list.append(
                self.whole_inference(imgs[i], img_metas[i], rescale))
        seg_logit = torch.stack(seg_logit_list).sum(dim=0)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        return seg_pred
