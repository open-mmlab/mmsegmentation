# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from mmcv.transforms import Compose

from mmseg.models import BaseSegmentor
from mmseg.registry import MODELS
from mmseg.utils import SampleList
from mmseg.visualization import SegLocalVisualizer


def init_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = MODELS.build(config.model)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def _preprare_data(imgs, model: BaseSegmentor):

    cfg = model.cfg
    if dict(type='LoadAnnotations') in cfg.test_pipeline:
        cfg.test_pipeline.remove(dict(type='LoadAnnotations'))

    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    if isinstance(imgs[0], np.ndarray):
        cfg.test_pipeline[0].type = 'LoadImageFromNDArray'

    # TODO: Consider using the singleton pattern to avoid building
    # a pipeline for each inference
    pipeline = Compose(cfg.test_pipeline)

    data = []
    for img in imgs:
        if isinstance(img, np.ndarray):
            data_ = dict(img=img)
        else:
            data_ = dict(img_path=img)
        data_ = pipeline(data_)
        data.append(data_)

    return data


def inference_model(model: BaseSegmentor, img):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    # prepare data
    data = _preprare_data(img, model)

    # forward the model
    with torch.no_grad():
        result = model.test_step(data)

    return result


def show_result_pyplot(model: BaseSegmentor,
                       img: Union[str, np.ndarray],
                       result: SampleList,
                       opacity: float = 0.5,
                       title: str = '',
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       wait_time: float = 0,
                       show: bool = True,
                       save_dir=None):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The prediction SegDataSample result.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        show (bool): Whether to display the drawn image.
            Default to True.
    """
    if hasattr(model, 'module'):
        model = model.module
    if isinstance(img, str):
        image = mmcv.imread(img)
    else:
        image = img
    if save_dir is not None:
        mmcv.mkdir_or_exist(save_dir)
    # init visualizer
    visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend')],
        save_dir=save_dir,
        alpha=opacity)
    visualizer.dataset_meta = dict(
        classes=model.CLASSES, palette=model.PALETTE)
    visualizer.add_datasample(
        name=title,
        image=image,
        pred_sample=result[0],
        draw_gt=draw_gt,
        draw_pred=draw_pred,
        wait_time=wait_time,
        show=show)
    return visualizer.get_image()
