# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import mkdir_or_exist, progressbar
from PIL import Image

from mmseg.registry import DATASETS

init_default_scope('mmseg')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix from segmentation results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test folder result')
    parser.add_argument(
        'save_dir', help='directory where confusion matrix will be saved')
    parser.add_argument(
        '--show', action='store_true', help='show confusion matrix')
    parser.add_argument(
        '--color-theme',
        default='winter',
        help='theme of the matrix color map')
    parser.add_argument(
        '--title',
        default='Normalized Confusion Matrix',
        help='title of the matrix color map')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def calculate_confusion_matrix(dataset, results):
    """Calculate the confusion matrix.

    Args:
        dataset (Dataset): Test or val dataset.
        results (list[ndarray]): A list of segmentation results in each image.
    """
    n = len(dataset.METAINFO['classes'])
    confusion_matrix = np.zeros(shape=[n, n])
    assert len(dataset) == len(results)
    ignore_index = dataset.ignore_index
    reduce_zero_label = dataset.reduce_zero_label
    prog_bar = progressbar.ProgressBar(len(results))
    for idx, per_img_res in enumerate(results):
        res_segm = per_img_res
        gt_segm = dataset[idx]['data_samples'] \
            .gt_sem_seg.data.squeeze().numpy().astype(np.uint8)
        gt_segm, res_segm = gt_segm.flatten(), res_segm.flatten()
        if reduce_zero_label:
            gt_segm = gt_segm - 1
        to_ignore = gt_segm == ignore_index

        gt_segm, res_segm = gt_segm[~to_ignore], res_segm[~to_ignore]
        inds = n * gt_segm + res_segm
        mat = np.bincount(inds, minlength=n**2).reshape(n, n)
        confusion_matrix += mat
        prog_bar.update()
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix,
                          labels,
                          save_dir=None,
                          show=True,
                          title='Normalized Confusion Matrix',
                          color_theme='OrRd'):
    """Draw confusion matrix with matplotlib.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_dir (str|optional): If set, save the confusion matrix plot to the
            given path. Default: None.
        show (bool): Whether to show the plot. Default: True.
        title (str): Title of the plot. Default: `Normalized Confusion Matrix`.
        color_theme (str): Theme of the matrix color map. Default: `winter`.
    """
    # normalize the confusion matrix
    per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    confusion_matrix = \
        confusion_matrix.astype(np.float32) / per_label_sums * 100

    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(2 * num_classes, 2 * num_classes * 0.8), dpi=300)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    colorbar = plt.colorbar(mappable=im, ax=ax)
    colorbar.ax.tick_params(labelsize=20)  # 设置 colorbar 标签的字体大小

    title_font = {'weight': 'bold', 'size': 20}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 40}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_yticklabels(labels, fontsize=20)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # draw confusion matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                '{}%'.format(
                    round(confusion_matrix[i, j], 2
                          ) if not np.isnan(confusion_matrix[i, j]) else -1),
                ha='center',
                va='center',
                color='k',
                size=20)

    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
    if save_dir is not None:
        mkdir_or_exist(save_dir)
        plt.savefig(
            os.path.join(save_dir, 'confusion_matrix.png'), format='png')
    if show:
        plt.show()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    results = []
    for img in sorted(os.listdir(args.prediction_path)):
        img = os.path.join(args.prediction_path, img)
        image = Image.open(img)
        image = np.copy(image)
        results.append(image)

    assert isinstance(results, list)
    if isinstance(results[0], np.ndarray):
        pass
    else:
        raise TypeError('invalid type of prediction results')

    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    confusion_matrix = calculate_confusion_matrix(dataset, results)
    plot_confusion_matrix(
        confusion_matrix,
        dataset.METAINFO['classes'],
        save_dir=args.save_dir,
        show=args.show,
        title=args.title,
        color_theme=args.color_theme)


if __name__ == '__main__':
    main()
