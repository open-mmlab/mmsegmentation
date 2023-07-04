# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence, Union

import mmcv
import numpy as np
import torch
from mmengine import Config
from mmengine.dataset import Compose
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.utils import mkdir_or_exist

from mmseg.models import BaseSegmentor
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import SampleList, dataset_aliases, get_classes, get_palette
from mmseg.visualization import SegLocalVisualizer


def init_model(config: Union[str, Path, Config],
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               cfg_options: Optional[dict] = None):
    """Initialize a segmentor from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
        cfg_options (dict, optional): Options to override some settings in
            the used config.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    config.model.pretrained = None
    config.model.train_cfg = None
    init_default_scope(config.get('default_scope', 'mmseg'))

    model = MODELS.build(config.model)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        dataset_meta = checkpoint['meta'].get('dataset_meta', None)
        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint.get('meta', {}):
            # mmseg 1.x
            model.dataset_meta = dataset_meta
        elif 'CLASSES' in checkpoint.get('meta', {}):
            # < mmseg 1.x
            classes = checkpoint['meta']['CLASSES']
            palette = checkpoint['meta']['PALETTE']
            model.dataset_meta = {'classes': classes, 'palette': palette}
        else:
            warnings.simplefilter('once')
            warnings.warn(
                'dataset_meta or class names are not saved in the '
                'checkpoint\'s meta data, classes and palette will be'
                'set according to num_classes ')
            num_classes = model.decode_head.num_classes
            dataset_name = None
            for name in dataset_aliases.keys():
                if len(get_classes(name)) == num_classes:
                    dataset_name = name
                    break
            if dataset_name is None:
                warnings.warn(
                    'No suitable dataset found, use Cityscapes by default')
                dataset_name = 'cityscapes'
            model.dataset_meta = {
                'classes': get_classes(dataset_name),
                'palette': get_palette(dataset_name)
            }
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


ImageType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def _preprare_data(imgs: ImageType, model: BaseSegmentor):

    cfg = model.cfg
    for t in cfg.test_pipeline:
        if t.get('type') == 'LoadAnnotations':
            cfg.test_pipeline.remove(t)

    is_batch = True
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
        is_batch = False

    if isinstance(imgs[0], np.ndarray):
        cfg.test_pipeline[0]['type'] = 'LoadImageFromNDArray'

    # TODO: Consider using the singleton pattern to avoid building
    # a pipeline for each inference
    pipeline = Compose(cfg.test_pipeline)

    data = defaultdict(list)
    for img in imgs:
        if isinstance(img, np.ndarray):
            data_ = dict(img=img)
        else:
            data_ = dict(img_path=img)
        data_ = pipeline(data_)
        data['inputs'].append(data_['inputs'])
        data['data_samples'].append(data_['data_samples'])

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
        will be returned, otherwise return the segmentation results directly.
    """
    # prepare data
    data, is_batch = _preprare_data(img, model)

    # forward the model
    with torch.no_grad():
        results = model.test_step(data)

    return results if is_batch else results[0]


def show_result_pyplot(model: BaseSegmentor,
                       img: Union[str, np.ndarray],
                       result: SegDataSample,
                       opacity: float = 0.5,
                       title: str = '',
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       wait_time: float = 0,
                       show: bool = True,
                       save_dir=None,
                       out_file=None):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (SegDataSample): The prediction SegDataSample result.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5. Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        draw_gt (bool): Whether to draw GT SegDataSample. Default to True.
        draw_pred (bool): Whether to draw Prediction SegDataSample.
            Defaults to True.
        wait_time (float): The interval of show (s). 0 is the special value
            that means "forever". Defaults to 0.
        show (bool): Whether to display the drawn image.
            Default to True.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        out_file (str, optional): Path to output file. Default to None.

    Returns:
        np.ndarray: the drawn image which channel is RGB.
    """
    if hasattr(model, 'module'):
        model = model.module
    if isinstance(img, str):
        image = mmcv.imread(img, channel_order='rgb')
    else:
        image = img
    if save_dir is not None:
        mkdir_or_exist(save_dir)
    # init visualizer
    visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend')],
        save_dir=save_dir,
        alpha=opacity)
    visualizer.dataset_meta = dict(
        classes=model.dataset_meta['classes'],
        palette=model.dataset_meta['palette'])
    visualizer.add_datasample(
        name=title,
        image=image,
        data_sample=result,
        draw_gt=draw_gt,
        draw_pred=draw_pred,
        wait_time=wait_time,
        out_file=out_file,
        show=show)
    vis_img = visualizer.get_image()

    return vis_img
