# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import mkdir_or_exist, progressbar
from mmengine.fileio import dump
from PIL import Image
from my_projects.conversion_tests.converters.dataset_converter import (
    DatasetConverter
)
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
        '--test_dataset',
        '-test_ds',
        type=str,
        default="HOTS",
        choices=[
            "HOTS", 
            "IRL_VISION",
            "HOTS_CAT", 
            "IRL_VISION_CAT",
            "ARID20",
            "ARID10",
            "ADE20K"
        ]
    )
    parser.add_argument(
        '--output_dataset',
        '-out_ds',
        type=str,
        default="HOTS",
        choices=[
            "HOTS", 
            "IRL_VISION",
            "HOTS_CAT", 
            "IRL_VISION_CAT",
            "ARID20",
            "ARID10",
            "ADE20K"
        ]
    )
    parser.add_argument(
        '--target_dataset',
        '-tar_ds',
        type=str,
        default="HOTS_CAT",
        choices=[
            "HOTS_CAT", 
            "IRL_VISION_CAT",
            "ARID20",
            "ARID10",
            "ADE20K"
        ]
    )
   
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


def calculate_confusion_matrix(
    dataset_converter: DatasetConverter, 
    test_dataset,
    results
):
    """Calculate the confusion matrix

    Args:
        dataset_converter (DatasetConverter): _description_
        test_dataset (_type_): _description_
        results (_type_): results are in target_dataset labels

    Returns:
        _type_: _description_
    """
    n = len(
        dataset_converter.get_class_names(
            dataset_name=dataset_converter.target_dataset
        )
    )
    confusion_matrix = np.zeros(shape=[n, n])
    assert len(test_dataset) == len(results)
    # TODO not sure if should be the test of target dataset
    ignore_index = dataset_converter.unknown_idx
    reduce_zero_label = test_dataset.reduce_zero_label
    prog_bar = progressbar.ProgressBar(len(results))
    for idx, per_img_res in enumerate(results):
        res_segm = per_img_res
        gt_segm = test_dataset[idx]['data_samples'] \
            .gt_sem_seg.data.squeeze().numpy().astype(np.uint8)
        
        gt_segm = dataset_converter.convert_gt_label(label=gt_segm)
        
        gt_segm, res_segm = gt_segm.flatten(), res_segm.flatten()
        if reduce_zero_label:
            gt_segm = gt_segm - 1
        to_ignore = gt_segm == ignore_index
        
        gt_segm, res_segm = gt_segm[~to_ignore], res_segm[~to_ignore]
        
        # inds = n * gt_segm + res_segm
        # print(np.unique(inds))
        # print(max(inds))
        # mat = np.bincount(inds, minlength=n**2).reshape(n, n)
        # print(np.unique(mat))
        
        # confusion_matrix += mat
        for gt_pix, pred_pix in zip(gt_segm, res_segm):
            if gt_pix == ignore_index or pred_pix == ignore_index:
                continue
            confusion_matrix[gt_pix][pred_pix] += 1
        prog_bar.update()
    return confusion_matrix




def dump_confusion_data(
    confusion_matrix, dataset_converter: DatasetConverter, 
    min_value = 0.0, save_dir=None,
    ignore_diagonal = True
):
    per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    confusion_matrix = \
        confusion_matrix.astype(np.float32) / per_label_sums * 100
    confusion_matrix = np.nan_to_num(confusion_matrix)
    conf_dict_list = []
    # row nr
    labels = dataset_converter.get_class_names(
        dataset_name=dataset_converter.target_dataset
    )
    for gt_label_idx, gt_label in enumerate(labels):
        
        for pred_label_idx, pred_label in enumerate(labels):
            if ignore_diagonal and gt_label_idx == pred_label_idx:
                continue
            conf_value = confusion_matrix[gt_label_idx][pred_label_idx]
            if conf_value < min_value:
                continue
            conf_dict_list.append(
                {
                    "gt_label"      :   gt_label,
                    "pred_label"    :   pred_label,
                    "score"         :   conf_value
                }
            )
    if save_dir is not None:
        mkdir_or_exist(save_dir)
        json_file = os.path.join(save_dir, "confusion.json")
        dump(conf_dict_list, json_file, indent=4)
        
def main():
    args = parse_args()
    # cfg should be of test_dataset 
    # predictions are in target
    # out_ds?
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    dataset_converter = DatasetConverter(
        test_dataset=args.test_dataset,
        output_dataset=args.output_dataset,
        target_dataset=args.target_dataset
    )
    
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

    test_dataset = DATASETS.build(cfg.test_dataloader.dataset)
    confusion_matrix = calculate_confusion_matrix(
        dataset_converter=dataset_converter, 
        test_dataset=test_dataset, 
        results=results
    )
    dump_confusion_data(
        confusion_matrix=confusion_matrix,
        dataset_converter=dataset_converter,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
