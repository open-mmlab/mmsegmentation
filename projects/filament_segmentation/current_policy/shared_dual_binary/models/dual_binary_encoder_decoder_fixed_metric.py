# Copyright (c) OpenMMLab.
# Dual binary segmentation with one shared DeepLabV3/UNet backbone.
#
# Place this file at:
#   mmseg/models/segmentors/dual_binary_encoder_decoder.py
#
# Original 3-class masks:
#   0 = background
#   1 = confirmed filament
#   2 = candidate filament
#   255 = ignore
#
# One shared backbone/neck is used, and only the decode head is branched:
#   shared features -> confirmed head (background vs confirmed)
#                   -> candidate head (background vs candidate)

from copy import deepcopy

import torch

from mmengine.evaluator import BaseMetric
from mmseg.models.segmentors import EncoderDecoder
from mmseg.registry import METRICS, MODELS


@MODELS.register_module()
class DualBinaryEncoderDecoder(EncoderDecoder):
    """EncoderDecoder with one shared feature extractor and two binary heads."""

    def __init__(
        self,
        *args,
        confirmed_label=1,
        candidate_label=2,
        ignore_index=255,
        confirmed_loss_weight=1.0,
        candidate_loss_weight=1.0,
        **kwargs,
    ):
        decode_head_cfg = deepcopy(kwargs.get('decode_head', None))
        if decode_head_cfg is None:
            raise ValueError('decode_head must be defined in the config.')

        super().__init__(*args, **kwargs)

        self.confirmed_head = self.decode_head
        self.candidate_head = MODELS.build(decode_head_cfg)

        self.confirmed_label = int(confirmed_label)
        self.candidate_label = int(candidate_label)
        self.ignore_index = int(ignore_index)
        self.confirmed_loss_weight = float(confirmed_loss_weight)
        self.candidate_loss_weight = float(candidate_loss_weight)

    def _make_binary_samples(self, batch_data_samples, positive_label):
        samples = deepcopy(batch_data_samples)
        for sample in samples:
            gt = sample.gt_sem_seg.data
            binary = torch.full_like(gt, fill_value=self.ignore_index)
            binary[gt == 0] = 0
            binary[gt == positive_label] = 1
            binary[gt == self.ignore_index] = self.ignore_index
            sample.gt_sem_seg.data = binary
        return samples

    @staticmethod
    def _prefix_losses(losses, prefix, weight):
        out = {}
        for key, value in losses.items():
            new_key = f'{prefix}.{key}'
            if 'loss' in key:
                if isinstance(value, (list, tuple)):
                    out[new_key] = [v * weight for v in value]
                else:
                    out[new_key] = value * weight
            else:
                out[new_key] = value
        return out

    def loss(self, inputs, data_samples):
        x = self.extract_feat(inputs)

        confirmed_samples = self._make_binary_samples(
            data_samples, self.confirmed_label
        )
        candidate_samples = self._make_binary_samples(
            data_samples, self.candidate_label
        )

        confirmed_losses = self.confirmed_head.loss(
            x, confirmed_samples, self.train_cfg
        )
        candidate_losses = self.candidate_head.loss(
            x, candidate_samples, self.train_cfg
        )

        losses = {}
        losses.update(
            self._prefix_losses(
                confirmed_losses, 'confirmed', self.confirmed_loss_weight
            )
        )
        losses.update(
            self._prefix_losses(
                candidate_losses, 'candidate', self.candidate_loss_weight
            )
        )
        return losses

    def _head_logits(self, x, batch_img_metas):
        confirmed_logits = self.confirmed_head.predict(
            x, batch_img_metas, self.test_cfg
        )
        candidate_logits = self.candidate_head.predict(
            x, batch_img_metas, self.test_cfg
        )
        return confirmed_logits, candidate_logits

    def predict(self, inputs, data_samples=None):
        if data_samples is not None:
            batch_img_metas = [s.metainfo for s in data_samples]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0],
                )
                for _ in range(inputs.shape[0])
            ]

        x = self.extract_feat(inputs)
        confirmed_logits, candidate_logits = self._head_logits(
            x, batch_img_metas
        )

        confirmed_processed = self.postprocess_result(
            confirmed_logits, deepcopy(data_samples)
        )
        candidate_processed = self.postprocess_result(
            candidate_logits, deepcopy(data_samples)
        )

        outputs = deepcopy(data_samples)
        for out, conf, cand in zip(
            outputs, confirmed_processed, candidate_processed
        ):
            out.pred_confirmed_sem_seg = conf.pred_sem_seg
            out.seg_logits_confirmed = conf.seg_logits
            out.pred_candidate_sem_seg = cand.pred_sem_seg
            out.seg_logits_candidate = cand.seg_logits

            # Compatibility fields for generic MMSeg utilities.
            out.pred_sem_seg = conf.pred_sem_seg
            out.seg_logits = conf.seg_logits

        return outputs

    def _forward(self, inputs, data_samples=None):
        x = self.extract_feat(inputs)
        confirmed_logits = self.confirmed_head.forward(x)
        candidate_logits = self.candidate_head.forward(x)
        return confirmed_logits, candidate_logits

    @torch.no_grad()
    def predict_both_logits(self, inputs, data_samples=None):
        if data_samples is not None:
            batch_img_metas = [s.metainfo for s in data_samples]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0],
                )
                for _ in range(inputs.shape[0])
            ]

        x = self.extract_feat(inputs)
        return self._head_logits(x, batch_img_metas)


@METRICS.register_module()
class DualBinarySegMetric(BaseMetric):
    """Validation metric for confirmed/candidate from one model."""

    def __init__(
        self,
        collect_device='cpu',
        prefix=None,
        confirmed_label=1,
        candidate_label=2,
        ignore_index=255,
    ):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.confirmed_label = int(confirmed_label)
        self.candidate_label = int(candidate_label)
        self.ignore_index = int(ignore_index)

    @staticmethod
    def _confusion(gt3, pred, positive_label, other_label, ignore_index):
        gt3 = gt3.reshape(-1).long()
        pred = pred.reshape(-1).long()

        valid = (gt3 != ignore_index) & (gt3 != other_label)
        gt = (gt3[valid] == positive_label).long()
        pr = pred[valid].long()

        encoded = 2 * gt + pr
        return torch.bincount(encoded, minlength=4).reshape(2, 2).cpu()

    @staticmethod
    def _get_tensor(sample, key):
        """Support both SegDataSample and dict input from MMEngine Evaluator."""
        if isinstance(sample, dict):
            value = sample[key]
        else:
            value = getattr(sample, key)

        if isinstance(value, dict):
            value = value['data']
        elif hasattr(value, 'data'):
            value = value.data

        return value

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            gt3 = (
                self._get_tensor(sample, 'gt_sem_seg')
                .squeeze(0)
                .cpu()
            )

            conf_pred = (
                self._get_tensor(sample, 'pred_confirmed_sem_seg')
                .squeeze(0)
                .cpu()
            )

            cand_pred = (
                self._get_tensor(sample, 'pred_candidate_sem_seg')
                .squeeze(0)
                .cpu()
            )

            conf_cm = self._confusion(
                gt3,
                conf_pred,
                self.confirmed_label,
                self.candidate_label,
                self.ignore_index,
            )
            cand_cm = self._confusion(
                gt3,
                cand_pred,
                self.candidate_label,
                self.confirmed_label,
                self.ignore_index,
            )

            self.results.append(
                dict(confirmed=conf_cm, candidate=cand_cm)
            )

    @staticmethod
    def _metrics_from_cm(cm):
        cm = cm.double()
        tn, fp = cm[0, 0], cm[0, 1]
        fn, tp = cm[1, 0], cm[1, 1]
        eps = 1e-12

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        f1 = 2 * tp / (2 * tp + fp + fn + eps)
        acc = (tp + tn) / (cm.sum() + eps)

        return dict(
            Precision=float(precision * 100.0),
            Recall=float(recall * 100.0),
            IoU=float(iou * 100.0),
            F1=float(f1 * 100.0),
            aAcc=float(acc * 100.0),
        )

    def compute_metrics(self, results):
        confirmed_cm = torch.zeros((2, 2), dtype=torch.int64)
        candidate_cm = torch.zeros((2, 2), dtype=torch.int64)

        for item in results:
            confirmed_cm += item['confirmed']
            candidate_cm += item['candidate']

        conf = self._metrics_from_cm(confirmed_cm)
        cand = self._metrics_from_cm(candidate_cm)

        return {
            'confirmed_Precision': conf['Precision'],
            'confirmed_Recall': conf['Recall'],
            'confirmed_IoU': conf['IoU'],
            'confirmed_F1': conf['F1'],
            'confirmed_aAcc': conf['aAcc'],
            'candidate_Precision': cand['Precision'],
            'candidate_Recall': cand['Recall'],
            'candidate_IoU': cand['IoU'],
            'candidate_F1': cand['F1'],
            'candidate_aAcc': cand['aAcc'],
        }
