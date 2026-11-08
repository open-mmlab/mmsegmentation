import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.logging import MMLogger
from mmengine.structures import PixelData
from mmseg.models import EncoderDecoder
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample


class OneHotCEDiceLoss(nn.Module):
    """
    2-channel one-hot loss for one binary segmentation head.

    Valid target:
        background -> class index 0 -> one-hot [1, 0]
        target     -> class index 1 -> one-hot [0, 1]
        ignore     -> 255

    CE is explicitly computed from the 2D one-hot target.
    Dice is computed on foreground channel 1.
    """

    def __init__(
        self,
        ce_weight=1.0,
        dice_weight=3.0,
        ignore_index=255,
        smooth=1.0,
    ):
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.ignore_index = int(ignore_index)
        self.smooth = float(smooth)

    def forward(self, logits, target):
        if logits.ndim != 4 or logits.shape[1] != 2:
            raise ValueError(
                f'Expected [N,2,H,W], got {tuple(logits.shape)}'
            )

        if target.ndim == 4 and target.shape[1] == 1:
            target = target[:, 0]

        target = target.long()
        valid = target != self.ignore_index

        if not torch.any(valid):
            return logits.sum() * 0.0

        safe_target = target.clone()
        safe_target[~valid] = 0

        one_hot = F.one_hot(
            safe_target.clamp(0, 1),
            num_classes=2,
        ).permute(0, 3, 1, 2).float()

        log_prob = F.log_softmax(
            logits,
            dim=1,
        )
        ce_map = -(
            one_hot * log_prob
        ).sum(dim=1)
        loss_ce = ce_map[valid].mean()

        prob_fg = F.softmax(
            logits,
            dim=1,
        )[:, 1]
        target_fg = one_hot[:, 1]

        p = prob_fg[valid]
        t = target_fg[valid]

        intersection = (p * t).sum()
        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            p.sum() + t.sum() + self.smooth
        )
        loss_dice = 1.0 - dice_score

        return (
            self.ce_weight * loss_ce
            + self.dice_weight * loss_dice
        )


@MODELS.register_module()
class HRFDeepLabV3UNetSharedASPPTwoHeadOneHot(
    EncoderDecoder
):
    """
    HRF-pretrained DeepLabV3 (UNet Backbone).

    Shared:
      UNet backbone
      ASPP branches
      ASPP bottleneck

    Separate only:
      Class 1 final classifier: Conv2d(16, 2, 1)
      Class 2 final classifier: Conv2d(16, 2, 1)

    Each head:
      channel 0 = background
      channel 1 = target filament class
    """

    def __init__(
        self,
        *args,
        pretrained_checkpoint=None,
        ce_weight=1.0,
        dice_weight=3.0,
        ignore_index=255,
        class1_threshold=0.5,
        class2_threshold=0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.pretrained_checkpoint = (
            pretrained_checkpoint
        )
        self.binary_ignore_index = int(
            ignore_index
        )
        self.class1_threshold = float(
            class1_threshold
        )
        self.class2_threshold = float(
            class2_threshold
        )

        shared_channels = int(
            self.decode_head.channels
        )

        self.class1_classifier = nn.Conv2d(
            shared_channels,
            2,
            kernel_size=1,
        )
        self.class2_classifier = nn.Conv2d(
            shared_channels,
            2,
            kernel_size=1,
        )

        self.class1_loss_fn = OneHotCEDiceLoss(
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            ignore_index=ignore_index,
        )
        self.class2_loss_fn = OneHotCEDiceLoss(
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            ignore_index=ignore_index,
        )

    def init_weights(self):
        super().init_weights()
        self._load_hrf_checkpoint()

        for classifier in (
            self.class1_classifier,
            self.class2_classifier,
        ):
            nn.init.normal_(
                classifier.weight,
                mean=0.0,
                std=0.01,
            )
            if classifier.bias is not None:
                nn.init.constant_(
                    classifier.bias,
                    0.0,
                )

        # HRF original final classifier is retained
        # only for checkpoint compatibility.
        for p in self.decode_head.conv_seg.parameters():
            p.requires_grad = False

        MMLogger.get_current_instance().info(
            'HRF pretrained weights loaded. '
            'ASPP shared; final heads are 16->2; '
            'training target is 2D one-hot per head.'
        )

    def _load_hrf_checkpoint(self):
        if self.pretrained_checkpoint is None:
            raise ValueError(
                'pretrained_checkpoint is required.'
            )

        if not os.path.isfile(
            self.pretrained_checkpoint
        ):
            raise FileNotFoundError(
                self.pretrained_checkpoint
            )

        checkpoint = torch.load(
            self.pretrained_checkpoint,
            map_location='cpu',
        )

        if (
            isinstance(checkpoint, dict)
            and 'state_dict' in checkpoint
        ):
            source_state = checkpoint[
                'state_dict'
            ]
        elif (
            isinstance(checkpoint, dict)
            and 'model' in checkpoint
        ):
            source_state = checkpoint['model']
        else:
            source_state = checkpoint

        cleaned = {}
        for key, value in source_state.items():
            if key.startswith('module.'):
                key = key[len('module.'):]
            cleaned[key] = value

        current = self.state_dict()

        compatible = {
            key: value
            for key, value in cleaned.items()
            if (
                key in current
                and current[key].shape == value.shape
            )
        }

        self.load_state_dict(
            compatible,
            strict=False,
        )

        logger = MMLogger.get_current_instance()
        logger.info(
            f'Loaded HRF checkpoint: '
            f'{self.pretrained_checkpoint}'
        )
        logger.info(
            f'Compatible tensors: '
            f'{len(compatible)}/{len(cleaned)}'
        )

    def _shared_aspp_feature(
        self,
        backbone_features,
    ):
        head = self.decode_head

        x = head._transform_inputs(
            backbone_features
        )

        image_pool = resize(
            input=head.image_pool(x),
            size=x.size()[2:],
            mode='bilinear',
            align_corners=head.align_corners,
        )

        aspp_outputs = [image_pool]
        aspp_outputs.extend(
            head.aspp_modules(x)
        )

        aspp_outputs = torch.cat(
            aspp_outputs,
            dim=1,
        )

        shared_feature = head.bottleneck(
            aspp_outputs
        )

        if head.dropout is not None:
            shared_feature = head.dropout(
                shared_feature
            )

        return shared_feature

    def _forward_dual(self, inputs):
        backbone_features = self.extract_feat(
            inputs
        )

        shared_feature = self._shared_aspp_feature(
            backbone_features
        )

        class1_logits = self.class1_classifier(
            shared_feature
        )
        class2_logits = self.class2_classifier(
            shared_feature
        )

        return class1_logits, class2_logits

    @staticmethod
    def _stack_gt(
        data_samples,
        device,
    ):
        gt = torch.stack(
            [
                sample.gt_sem_seg.data.squeeze(0)
                for sample in data_samples
            ],
            dim=0,
        )
        return gt.to(device)

    def loss(
        self,
        inputs,
        data_samples,
    ):
        class1_logits, class2_logits = (
            self._forward_dual(inputs)
        )

        gt = self._stack_gt(
            data_samples,
            class1_logits.device,
        )

        # Head 1:
        # 0 -> [1,0]
        # 1 -> [0,1]
        # 2 -> ignore
        class1_target = torch.full_like(
            gt,
            self.binary_ignore_index,
        )
        class1_target[gt == 0] = 0
        class1_target[gt == 1] = 1

        # Head 2:
        # 0 -> [1,0]
        # 2 -> [0,1]
        # 1 -> ignore
        class2_target = torch.full_like(
            gt,
            self.binary_ignore_index,
        )
        class2_target[gt == 0] = 0
        class2_target[gt == 2] = 1

        loss1 = self.class1_loss_fn(
            class1_logits,
            class1_target,
        )
        loss2 = self.class2_loss_fn(
            class2_logits,
            class2_target,
        )

        return dict(
            loss_class1_onehot_ce_dice=loss1,
            loss_class2_onehot_ce_dice=loss2,
        )

    def _forward(
        self,
        inputs,
        data_samples=None,
    ):
        return self._forward_dual(inputs)

    def predict(
        self,
        inputs,
        data_samples=None,
    ):
        class1_logits, class2_logits = (
            self._forward_dual(inputs)
        )

        class1_prob_2ch = F.softmax(
            class1_logits,
            dim=1,
        )
        class2_prob_2ch = F.softmax(
            class2_logits,
            dim=1,
        )

        p1 = class1_prob_2ch[:, 1]
        p2 = class2_prob_2ch[:, 1]

        pred = torch.zeros_like(
            p1,
            dtype=torch.long,
        )

        m1 = p1 >= self.class1_threshold
        m2 = p2 >= self.class2_threshold

        pred[m1 & ~m2] = 1
        pred[m2 & ~m1] = 2

        both = m1 & m2
        pred[
            both & (p1 >= p2)
        ] = 1
        pred[
            both & (p2 > p1)
        ] = 2

        results = []

        for i in range(pred.shape[0]):
            sample = (
                data_samples[i]
                if data_samples is not None
                else SegDataSample()
            )

            sample.pred_sem_seg = PixelData(
                data=pred[i].unsqueeze(0)
            )

            # Foreground probability used by
            # PR/Precision/Recall scripts.
            sample.class1_prob = PixelData(
                data=p1[i].unsqueeze(0)
            )
            sample.class2_prob = PixelData(
                data=p2[i].unsqueeze(0)
            )

            # Full 2D softmax vectors.
            sample.class1_prob_2ch = PixelData(
                data=class1_prob_2ch[i]
            )
            sample.class2_prob_2ch = PixelData(
                data=class2_prob_2ch[i]
            )

            results.append(sample)

        return results
