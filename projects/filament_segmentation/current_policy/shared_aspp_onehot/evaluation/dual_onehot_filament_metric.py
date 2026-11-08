import numpy as np
import torch

from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS


def _field(sample, name):
    if isinstance(sample, dict):
        return sample.get(name, None)
    return getattr(sample, name, None)


def _pixel_data_to_tensor(x):
    if x is None:
        return None

    if isinstance(x, dict):
        x = x.get('data', x)

    if hasattr(x, 'data') and not torch.is_tensor(x):
        x = x.data

    if torch.is_tensor(x):
        return x

    return torch.as_tensor(x)


@METRICS.register_module()
class DualOneHotFilamentMetric(BaseMetric):
    """
    Independent binary metrics for the two 2-channel one-hot heads.

    Class 1:
        evaluate only original labels 0 and 1
    Class 2:
        evaluate only original labels 0 and 2
    """

    def __init__(
        self,
        class1_threshold=0.5,
        class2_threshold=0.5,
        ignore_index=255,
        collect_device='cpu',
        prefix=None,
    ):
        super().__init__(
            collect_device=collect_device,
            prefix=prefix,
        )

        self.class1_threshold = float(
            class1_threshold
        )
        self.class2_threshold = float(
            class2_threshold
        )
        self.ignore_index = int(
            ignore_index
        )

    def process(
        self,
        data_batch,
        data_samples,
    ):
        for sample in data_samples:
            gt_obj = _field(
                sample,
                'gt_sem_seg',
            )
            p1_obj = _field(
                sample,
                'class1_prob',
            )
            p2_obj = _field(
                sample,
                'class2_prob',
            )

            gt = _pixel_data_to_tensor(
                gt_obj
            )
            p1 = _pixel_data_to_tensor(
                p1_obj
            )
            p2 = _pixel_data_to_tensor(
                p2_obj
            )

            if (
                gt is None
                or p1 is None
                or p2 is None
            ):
                raise RuntimeError(
                    'gt_sem_seg, class1_prob and '
                    'class2_prob are required.'
                )

            gt = gt.squeeze().long().cpu()
            p1 = p1.squeeze().float().cpu()
            p2 = p2.squeeze().float().cpu()

            valid1 = (
                (gt == 0)
                | (gt == 1)
            )
            valid2 = (
                (gt == 0)
                | (gt == 2)
            )

            y1 = (
                gt[valid1] == 1
            )
            y2 = (
                gt[valid2] == 2
            )

            pred1 = (
                p1[valid1]
                >= self.class1_threshold
            )
            pred2 = (
                p2[valid2]
                >= self.class2_threshold
            )

            self.results.append(
                dict(
                    c1_tp=int(
                        (pred1 & y1).sum()
                    ),
                    c1_fp=int(
                        (pred1 & ~y1).sum()
                    ),
                    c1_fn=int(
                        (~pred1 & y1).sum()
                    ),
                    c1_tn=int(
                        (~pred1 & ~y1).sum()
                    ),
                    c1_gt=int(
                        y1.sum()
                    ),
                    c1_pred=int(
                        pred1.sum()
                    ),
                    c1_max=float(
                        p1[valid1].max()
                    ) if valid1.any() else 0.0,
                    c1_sum=float(
                        p1[valid1].sum()
                    ),
                    c1_n=int(
                        valid1.sum()
                    ),

                    c2_tp=int(
                        (pred2 & y2).sum()
                    ),
                    c2_fp=int(
                        (pred2 & ~y2).sum()
                    ),
                    c2_fn=int(
                        (~pred2 & y2).sum()
                    ),
                    c2_tn=int(
                        (~pred2 & ~y2).sum()
                    ),
                    c2_gt=int(
                        y2.sum()
                    ),
                    c2_pred=int(
                        pred2.sum()
                    ),
                    c2_max=float(
                        p2[valid2].max()
                    ) if valid2.any() else 0.0,
                    c2_sum=float(
                        p2[valid2].sum()
                    ),
                    c2_n=int(
                        valid2.sum()
                    ),
                )
            )

    @staticmethod
    def _metrics(
        tp,
        fp,
        fn,
    ):
        eps = 1e-12

        precision = (
            tp
            / (tp + fp + eps)
        )
        recall = (
            tp
            / (tp + fn + eps)
        )
        iou = (
            tp
            / (tp + fp + fn + eps)
        )
        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
                + eps
            )
        )

        return (
            precision,
            recall,
            iou,
            f1,
        )

    def compute_metrics(
        self,
        results,
    ):
        def s(key):
            return sum(
                x[key]
                for x in results
            )

        out = {}

        for prefix in ('c1', 'c2'):
            tp = s(f'{prefix}_tp')
            fp = s(f'{prefix}_fp')
            fn = s(f'{prefix}_fn')

            precision, recall, iou, f1 = (
                self._metrics(
                    tp,
                    fp,
                    fn,
                )
            )

            name = (
                'Class1'
                if prefix == 'c1'
                else 'Class2'
            )

            n = s(f'{prefix}_n')
            mean_prob = (
                s(f'{prefix}_sum')
                / max(n, 1)
            )

            out[
                f'{name}_Precision'
            ] = precision * 100.0
            out[
                f'{name}_Recall'
            ] = recall * 100.0
            out[
                f'{name}_IoU'
            ] = iou * 100.0
            out[
                f'{name}_F1'
            ] = f1 * 100.0
            out[
                f'{name}_Dice'
            ] = f1 * 100.0
            out[
                f'{name}_PredPositivePixels'
            ] = s(f'{prefix}_pred')
            out[
                f'{name}_GTPositivePixels'
            ] = s(f'{prefix}_gt')
            out[
                f'{name}_MaxProbability'
            ] = max(
                x[f'{prefix}_max']
                for x in results
            )
            out[
                f'{name}_MeanProbability'
            ] = mean_prob

        return out
