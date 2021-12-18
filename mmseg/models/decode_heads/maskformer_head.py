import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, build_plugin_layer, kaiming_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import force_fp32

from mmseg.core import multi_apply, reduce_mean

from ..builder import HEADS, build_loss, build_assigner, build_transformer
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class MaskFormerHead(BaseDecodeHead):
    """Implements the MaskFormer head.

    See `paper: Per-Pixel Classification is Not All You Need
    for Semantic Segmentation<https://arxiv.org/pdf/2107.06278>`
    for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number channels for feature.
        out_channels (int): Number channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer.
        pixel_decoder (obj:`mmcv.ConfigDict`|dict): Config for pixel decoder.
            Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add a layer
            to change the embed_dim of tranformer encoder in pixel decoder to
            the embed_dim of transformer decoder. Defaults to False.
        transformer_decoder (obj:`mmcv.ConfigDict`|dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (obj:`mmcv.ConfigDict`|dict): Config for
            transformer decoder position encoding. Defaults to None.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the classification
            loss. Defaults to `CrossEntropyLoss`.
        loss_mask (obj:`mmcv.ConfigDict`|dict): Config of the mask loss.
            Defaults to `FocalLoss`.
        loss_dice (obj:`mmcv.ConfigDict`|dict): Config of the dice loss.
            Defaults to `DiceLoss`.
        train_cfg (obj:`mmcv.ConfigDict`|dict): Training config of Maskformer
            head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of Maskformer
            head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 out_channels,
                 num_queries=100,
                 pixel_decoder=None,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_mask=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=20.0),
                 loss_dice=dict(
                     type='DiceLoss',
                     use_sigmoid=True,
                     activate=True,
                     naive_dice=True,
                     loss_weight=1.0),
                 assigner=dict(
                     type='MaskHungarianAssigner',
                     cls_cost=dict(type='ClassificationCost', weight=1.),
                     dice_cost=dict(type='DiceCost', weight=1.0, pred_act=True,
                                    eps=1.0),
                     mask_cost=dict(type='MaskFocalLossCost', weight=20.0)),
                 **kwargs):
        super(MaskFormerHead, self).__init__(input_transform='multiple_select',
                                             **kwargs)
        self.num_queries = num_queries

        pixel_decoder.update(
            in_channels=self.in_channels,
            feat_channels=self.channels,
            out_channels=out_channels)
        self.pixel_decoder = build_plugin_layer(pixel_decoder)[1]
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims
        pixel_decoder_type = pixel_decoder.get('type')
        if pixel_decoder_type == 'PixelDecoder' and (
                self.decoder_embed_dims != self.in_channels[-1]
                or enforce_decoder_input_project):
            self.decoder_input_proj = Conv2d(
                self.in_channels[-1], self.decoder_embed_dims, kernel_size=1)
        else:
            self.decoder_input_proj = nn.Identity()
        self.decoder_pe = build_positional_encoding(positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, out_channels)

        self.cls_embed = nn.Linear(self.channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(self.channels, self.channels), nn.ReLU(inplace=True),
            nn.Linear(self.channels, self.channels), nn.ReLU(inplace=True),
            nn.Linear(self.channels, out_channels))

        self.assigner = build_assigner(assigner)

        self.bg_cls_weight = 0
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is MaskFormerHead):
            assert isinstance(class_weight, float), 'Expected ' \
                                                    'class_weight to have type float. Found ' \
                                                    f'{type(class_weight)}.'
            # NOTE following the official MaskFormerHead repo, bg_cls_weight
            # means relative classification weight of the VOID class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                                                     'bg_cls_weight to have type float. Found ' \
                                                     f'{type(bg_cls_weight)}.'
            class_weight = (self.num_classes + 1) * [class_weight]
            # set VOID class as the last indice
            class_weight[self.num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
            'The classification weight for loss and matcher should be' \
            'exactly the same.'
        assert loss_dice['loss_weight'] == assigner['dice_cost']['weight'], \
            f'The dice weight for loss and matcher' \
            f'should be exactly the same.'
        assert loss_mask['loss_weight'] == assigner['mask_cost']['weight'], \
            'The focal weight for loss and matcher should be' \
            'exactly the same.'
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

        self.init_weights()

    def init_weights(self):
        kaiming_init(self.decoder_input_proj, a=1)

    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                    gt_masks_list, img_metas):
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape [num_queries,
                cls_out_channels].
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape [num_queries, h, w].
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels of all images.
                    Each with shape [num_queries, ].
                - label_weights_list (list[Tensor]): Label weights of all
                    images.Each with shape [num_queries, ].
                - mask_targets_list (list[Tensor]): Mask targets of all images.
                    Each with shape [num_queries, h, w].
                - mask_weights_list (list[Tensor]): Mask weights of all images.
                    Each with shape [num_queries, ].
                - num_total_pos (int): Number of positive samples in all
                    images.
                - num_total_neg (int): Number of negative samples in all
                    images.
        """
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list,
                                      mask_preds_list, gt_labels_list,
                                      gt_masks_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks,
                           img_metas):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape [num_queries, h, w].
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (n, ). n is the sum of number of stuff type and number
                of instance in a image.
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (n, h, w).
            img_metas (dict): Image informtation.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                    shape [num_queries, ].
                - label_weights (Tensor): Label weights of each image.
                    shape [num_queries, ].
                - mask_targets (Tensor): Mask targets of each image.
                    shape [num_queries, h, w].
                - mask_weights (Tensor): Mask weights of each image.
                    shape [num_queries, ].
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        target_shape = mask_pred.shape[-2:]
        gt_masks_downsampled = F.interpolate(
            gt_masks.unsqueeze(1).float(), target_shape,
            mode='nearest').squeeze(1).long()
        # assign and sample
        assign_result = self.assigner.assign(cls_score, mask_pred, gt_labels,
                                             gt_masks_downsampled, img_metas)
        # pos_ind: range from 1 to (self.num_classes)
        # which represents the positive index
        pos_inds = torch.nonzero(assign_result.gt_inds > 0,
                                 as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0,
                                 as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        # label target
        labels = gt_labels.new_full((self.num_queries,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones(self.num_queries)

        # mask target
        mask_targets = gt_masks[pos_assigned_gt_inds, :]
        mask_weights = mask_pred.new_zeros((self.num_queries,))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds)

    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss(self, all_cls_scores, all_mask_preds, gt_labels_list,
             gt_masks_list, img_metas):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape [num_decoder, batch_size, num_queries,
                cls_out_channels].
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape [num_decoder, batch_size, num_queries, h, w].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (n, ). n is the sum of number of stuff type
                and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image with
                shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice = multi_apply(
            self.loss_single, all_cls_scores, all_mask_preds,
            all_gt_labels_list, all_gt_masks_list, img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self, cls_scores, mask_preds, gt_labels_list,
                    gt_masks_list, img_metas):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape [batch_size, num_queries,
                cls_out_channels].
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape [batch_size, num_queries, h, w].
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (n, ). n is the sum of number of stuff
                types and number of instances in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]:Loss components for outputs from a single decoder
                layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,
         num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list,
                                           gt_labels_list, gt_masks_list,
                                           img_metas)
        # shape [batch_size, num_queries]
        labels = torch.stack(labels_list, dim=0)
        # shape [batch_size, num_queries]
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape [num_gts, h, w]
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape [batch_size, num_queries]
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape [batch_size * num_queries, ]
        cls_scores = cls_scores.flatten(0, 1)
        # shape [batch_size * num_queries, ]
        labels = labels.flatten(0, 1)
        # shape [batch_size* num_queries, ]
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_ones(self.num_classes + 1)
        class_weight[-1] = self.bg_cls_weight

        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        mask_preds = mask_preds[mask_weights > 0]
        target_shape = mask_targets.shape[-2:]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        # upsample to shape of target
        # shape [num_gts, h, w]
        mask_preds = F.interpolate(
            mask_preds.unsqueeze(1),
            target_shape,
            mode='bilinear',
            align_corners=False).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_preds, mask_targets, avg_factor=num_total_masks)

        # mask loss
        # FocalLoss support input of shape [n, num_class]
        h, w = mask_preds.shape[-2:]
        # shape [num_gts, h, w] -> [num_gts * h * w, 1]
        mask_preds = mask_preds.reshape(-1, 1)
        # shape [num_gts, h, w] -> [num_gts * h * w]
        mask_targets = mask_targets.reshape(-1)
        # target is (1 - mask_targets) !!!
        loss_mask = self.loss_mask(
            mask_preds, 1 - mask_targets, avg_factor=num_total_masks * h * w)

        return loss_cls, loss_mask, loss_dice

    def forward(self, feats, img_metas):
        """Forward function.

        Args:
            feats (list[Tensor]): Features from the upstream network, each
                is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Classification scores for each
                scale level. Each is a 4D-tensor with shape
                [num_decoder, batch_size, num_queries, cls_out_channels].
                 Note `cls_out_channels` should includes background.
            all_mask_preds (Tensor): Mask scores for each decoder
                layer. Each with shape [num_decoder, batch_size,
                num_queries, h, w].
        """
        batch_size = len(img_metas)
        input_img_h, input_img_w = img_metas[0]['pad_shape'][:-1]
        # input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        padding_mask = feats[-1].new_ones(
            (batch_size, input_img_h, input_img_w), dtype=torch.float32)
        for i in range(batch_size):
            img_h, img_w, _ = img_metas[i]['img_shape']
            padding_mask[i, :img_h, :img_w] = 0
        padding_mask = F.interpolate(
            padding_mask.unsqueeze(1),
            size=feats[-1].shape[-2:],
            mode='nearest').to(torch.bool).squeeze(1)
        # when backbone is swin, memory is output of last stage of swin.
        # when backbone is r50, memory is output of tranformer encoder.
        mask_features, memory = self.pixel_decoder(feats, img_metas)
        pos_embed = self.decoder_pe(padding_mask)
        memory = self.decoder_input_proj(memory)
        # shape [batch_size, c, h, w] -> [h*w, batch_size, c]
        memory = memory.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # shape [batch_size, h * w]
        padding_mask = padding_mask.flatten(1)
        # shape = [num_queries, embed_dims]
        query_embed = self.query_embed.weight
        # shape = [num_queries, batch_size, embed_dims]
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        target = torch.zeros_like(query_embed)
        # shape [num_decoder, num_queries, batch_size, embed_dims]
        out_dec = self.transformer_decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=padding_mask)
        # shape [num_decoder, batch_size, num_queries, embed_dims]
        out_dec = out_dec.transpose(1, 2)

        # cls_scores
        all_cls_scores = self.cls_embed(out_dec)

        # mask_preds
        mask_embed = self.mask_embed(out_dec)
        all_mask_preds = torch.einsum('lbqc,bchw->lbqhw', mask_embed,
                                      mask_features)

        return all_cls_scores, all_mask_preds

    def forward_train(self,
                      x,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      *,
                      gt_labels,
                      gt_masks):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Multi-level features from the upstream network,
                each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_semantic_seg (list[tensor]):Each element is the ground truth
                of semantic segmentation with the shape (N, H, W).
            train_cfg (dict): The training config, which not been used in
                maskformer.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, h, w).

        Returns:
            losses (dict[str, Tensor]): a dictionary of loss components
        """

        # forward
        all_cls_scores, all_mask_preds = self(x, img_metas)

        # loss
        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks,
                           img_metas)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Test segment without test-time aumengtation.

        Only the output of last decoder layers was used.

        Args:
            inputs (list[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            test_cfg (dict): Testing config.

        Returns:
            seg_mask (Tensor): Predicted semantic segmentation logits.
        """
        all_cls_scores, all_mask_preds = self(inputs, img_metas)
        cls_score, mask_pred = all_cls_scores[-1], all_mask_preds[-1]
        ori_h, ori_w, _ = img_metas[0]['ori_shape']

        # semantic inference
        cls_score = F.softmax(cls_score, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        seg_mask = torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred)
        return seg_mask
