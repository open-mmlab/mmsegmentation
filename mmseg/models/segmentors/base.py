# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16
from mmengine.data import PixelData

from mmseg.core import SegDataSample
from mmseg.core.utils import stack_batch


class BaseSegmentor(BaseModule, metaclass=ABCMeta):
    """Base class for segmentors.

    Args:
        preprocess_cfg (dict, optional): Model preprocessing config
            for processing the input data. it usually includes
            ``to_rgb``, ``pad_size_divisor``, ``pad_value``,
            ``mean`` and ``std``. Default to None.
       init_cfg (dict, optional): the config to control the
           initialization. Default to None.
    """

    def __init__(self, preprocess_cfg=None, init_cfg=None):
        super(BaseSegmentor, self).__init__(init_cfg)
        self.fp16_enabled = False
        self.preprocess_cfg = preprocess_cfg

        self.pad_value = 0

        if self.preprocess_cfg is not None:
            assert isinstance(self.preprocess_cfg, dict)
            self.preprocess_cfg = copy.deepcopy(self.preprocess_cfg)

            self.to_rgb = preprocess_cfg.get('to_rgb', False)
            self.pad_value = preprocess_cfg.get('pad_value', 0)
            self.size = preprocess_cfg.get('size')
            self.seg_pad_val = preprocess_cfg.get('seg_pad_val', 255)

            self.register_buffer(
                'pixel_mean',
                torch.tensor(preprocess_cfg['mean']).view(-1, 1, 1), False)
            self.register_buffer(
                'pixel_std',
                torch.tensor(preprocess_cfg['std']).view(-1, 1, 1), False)
        else:
            # Only used to provide device information
            self.register_buffer('pixel_mean', torch.tensor(1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    @property
    def with_neck(self):
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_auxiliary_head(self):
        """bool: whether the segmentor has auxiliary head"""
        return hasattr(self,
                       'auxiliary_head') and self.auxiliary_head is not None

    @property
    def with_decode_head(self):
        """bool: whether the segmentor has decode head"""
        return hasattr(self, 'decode_head') and self.decode_head is not None

    @abstractmethod
    def extract_feat(self, batch_inputs):
        """Placeholder for extract features from images."""
        pass

    @abstractmethod
    def encode_decode(self, batch_inputs, batch_data_samples):
        """Placeholder for encode images with backbone and decode into a
        semantic segmentation map of the same size as input."""
        pass

    @auto_fp16(apply_to=('batch_inputs', ))
    def forward_train(self, batch_inputs, batch_data_samples, **kwargs):
        """Placeholder for Forward function for training."""
        pass

    @abstractmethod
    def simple_test(self, batch_inputs, batch_img_metas, **kwargs):
        """Placeholder for single image test."""
        pass

    @abstractmethod
    def aug_test(self, batch_inputs, batch_img_metas, **kwargs):
        """Placeholder for augmentation test."""
        pass

    @auto_fp16(apply_to=('batch_inputs', ))
    def forward_test(self, batch_inputs, batch_data_samples, **kwargs):
        """
        Args:
            batch_inputs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `batch_img_metas`.
        """
        batch_size = len(batch_data_samples)
        batch_img_metas = []
        for batch_index in range(batch_size):
            metainfo = batch_data_samples[batch_index].metainfo
            metainfo['batch_input_shape'] = \
                tuple(batch_inputs[batch_index].size()[-2:])
            batch_img_metas.append(metainfo)

        # TODO: support aug_test
        num_augs = 1
        if num_augs == 1:
            return self.simple_test(
                torch.unsqueeze(batch_inputs[0], 0), batch_img_metas, **kwargs)
        else:
            # TODO: refactor and support aug test later
            return self.aug_test(batch_inputs, batch_img_metas, **kwargs)

    def forward(self, data, return_loss=False, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Args:
            data (list[dict]): The output of dataloader.
            return_loss (bool): Whether to return loss. In general,
                it will be set to True during training and False
                during testing. Default to False.

        Returns:
            during training
                dict: It should contain at least 3 keys: ``loss``,
                ``log_vars``, ``num_samples``.
                    - ``loss`` is a tensor for back propagation, which can be a
                      weighted sum of multiple losses.
                    - ``log_vars`` contains all the variables to be sent to the
                        logger.
                    - ``num_samples`` indicates the batch size (when the model
                        is DDP, it means the batch size on each GPU), which is
                        used for averaging the logs.

            during testing
                list[np.ndarray]: The predicted value obtained.
        """
        batch_inputs, batch_data_samples = self.preprocss_data(
            data, return_loss)
        if return_loss:
            losses = self.forward_train(batch_inputs, batch_data_samples,
                                        **kwargs)
            loss, log_vars = self._parse_losses(losses)

            outputs = dict(
                loss=loss,
                log_vars=log_vars,
                num_samples=len(batch_data_samples))
            return outputs
        else:
            return self.forward_test(batch_inputs, batch_data_samples,
                                     **kwargs)

    def preprocss_data(self, data, return_loss):
        """ Process input data during training and simple testing phases.
        Args:
            data (list[dict]): The data to be processed, which
                comes from dataloader.
            return_loss (bool): Train or test.

        Returns:
            tuple:  It should contain 2 item.
                 - batch_inputs (Tensor): The batch input tensor.
                 - batch_data_samples (list[:obj:`SegDataSample`]): The Data
                     Samples. It usually includes information such as
                     `gt_sem_seg`.
        """
        inputs = [data_['inputs'] for data_ in data]
        data_samples = [data_['data_sample'] for data_ in data]

        batch_data_samples = [
            data_sample.to(self.device) for data_sample in data_samples
        ]
        inputs = [_input.to(self.device) for _input in inputs]

        if self.preprocess_cfg is None:
            batch_inputs, batch_data_samples = stack_batch(
                inputs, batch_data_samples)
            return batch_inputs.float(), batch_data_samples

        if self.to_rgb and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]
        batch_inputs = [(_input - self.pixel_mean) / self.pixel_std
                        for _input in inputs]
        if return_loss:
            batch_inputs, batch_data_samples = stack_batch(
                batch_inputs, batch_data_samples, self.size, self.pad_value,
                self.seg_pad_val)
        return batch_inputs, batch_data_samples

    def postprocess_result(self, results_dict: dict) -> list:
        """ Convert results list to `SegDataSample`.
        Args:
            results_dict (dict): Segmentation results of
                each image. It usually contain 'seg_logits' and 'pred_sem_seg'

        Returns:
            dict: Segmentation results of the input images.
                It usually contain 'seg_logits' and 'pred_sem_seg'.
        """
        batch_datasampes = [
            SegDataSample()
            for _ in range(results_dict['pred_sem_seg'].shape[0])
        ]
        for key, value in results_dict.items():
            for i in range(value.shape[0]):
                batch_datasampes[i].set_data({key: PixelData(data=value[i])})
        return batch_datasampes

    def train_step(self, data_batch, optim_wrapper, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(data_batch, True)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']))

        return outputs

    def val_step(self, data_batch, optim_wrapper=None, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        log_vars_ = dict()
        for loss_name, loss_value in log_vars.items():
            k = loss_name + '_val'
            log_vars_[k] = loss_value

        outputs = dict(
            loss=loss,
            log_vars=log_vars_,
            num_samples=len(data_batch['img_metas']))

        return outputs

    def test_step(self, data_batch):
        """The iteration step during test."""
        predictions = self(data_batch)

        return predictions

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, raise assertion error
        # to prevent GPUs from infinite waiting.
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()) + '\n')
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def show_result(self,
                    img,
                    result,
                    palette=None,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    opacity=0.5):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        seg = result[0]
        if palette is None:
            if self.PALETTE is None:
                # Get random state before set seed,
                # and restore random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(
                    0, 255, size=(len(self.CLASSES), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE
        palette = np.array(palette)
        assert palette.shape[0] == len(self.CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img
