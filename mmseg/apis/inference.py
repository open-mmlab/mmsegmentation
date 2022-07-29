# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union

import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from mmengine import Config
from mmengine.dataset import Compose

from mmseg.data import SegDataSample
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
        config = Config.fromfile(config)
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


ImageType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def _preprare_data(imgs: ImageType, model: BaseSegmentor):

    cfg = model.cfg
    if dict(type='LoadAnnotations') in cfg.test_pipeline:
        cfg.test_pipeline.remove(dict(type='LoadAnnotations'))

    is_batch = True
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
        is_batch = False

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

    return data, is_batch


def inference_model(model: BaseSegmentor,
                    img: ImageType) -> Union[SegDataSample, SampleList]:
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        :obj:`SegDataSample` or list[:obj:`SegDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """
    # prepare data
    data, is_batch = _preprare_data(img, model)

    # forward the model
    with torch.no_grad():
        results = model.test_step(data)

    return results if is_batch else results[0]


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
            Default 0.5. Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        draw_gt (bool): Whether to draw GT SegDataSample. Default to True.
        draw_pred (bool): Whether to draw Prediction SegDataSample.
            Defaults to True.
        wait_time (float): The interval of show (s). Defaults to 0.
        show (bool): Whether to display the drawn image.
            Default to True.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
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
