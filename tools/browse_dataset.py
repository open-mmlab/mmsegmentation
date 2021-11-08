import argparse
import os
import warnings
from pathlib import Path

import mmcv
import numpy as np
from mmcv import Config

from mmseg.datasets.builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--show-origin',
        default=False,
        action='store_true',
        help='if True, omit all augmentation in pipeline,'
        ' show origin image and seg map')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline，if `show-origin` is true, '
        'all pipeline except `Load` will be skipped')
    parser.add_argument(
        '--output-dir',
        default='./output',
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=int,
        default=999,
        help='the interval of show (ms)')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='the opacity of semantic map')
    args = parser.parse_args()
    return args


def imshow_semantic(img,
                    seg,
                    class_names,
                    palette=None,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    opacity=0.5):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        seg (Tensor): The semantic segmentation results to draw over
            `img`.
        class_names (list[str]): Names of each classes.
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
    if palette is None:
        palette = np.random.randint(0, 255, size=(len(class_names), 3))
    palette = np.array(palette)
    assert palette.shape[0] == len(class_names)
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


def _retrieve_data_cfg(_data_cfg, skip_type, show_origin):
    if show_origin is True:
        # only keep pipeline of Loading data and ann
        _data_cfg['pipeline'] = [
            x for x in _data_cfg.pipeline if 'Load' in x['type']
        ]
    else:
        _data_cfg['pipeline'] = [
            x for x in _data_cfg.pipeline if x['type'] not in skip_type
        ]


def retrieve_data_cfg(config_path, skip_type, show_origin=False):
    cfg = Config.fromfile(config_path)
    train_data_cfg = cfg.data.train
    if isinstance(train_data_cfg, list):
        for _data_cfg in train_data_cfg:
            if 'pipeline' in _data_cfg:
                _retrieve_data_cfg(_data_cfg, skip_type, show_origin)
            elif 'dataset' in _data_cfg:
                _retrieve_data_cfg(_data_cfg['dataset'], skip_type,
                                   show_origin)
            else:
                raise ValueError
    elif 'dataset' in train_data_cfg:
        _retrieve_data_cfg(train_data_cfg['dataset'], skip_type, show_origin)
    else:
        _retrieve_data_cfg(train_data_cfg, skip_type, show_origin)
    return cfg


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.show_origin)
    dataset = build_dataset(cfg.data.train)
    progress_bar = mmcv.ProgressBar(len(dataset))
    for item in dataset:
        filename = os.path.join(args.output_dir,
                                Path(item['filename']).name
                                ) if args.output_dir is not None else None
        imshow_semantic(
            item['img'],
            item['gt_semantic_seg'],
            dataset.CLASSES,
            dataset.PALETTE,
            show=args.show,
            wait_time=args.show_interval,
            out_file=filename,
            opacity=args.opacity,
        )
        progress_bar.update()


if __name__ == '__main__':
    main()
