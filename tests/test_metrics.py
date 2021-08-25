# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmseg.core.evaluation import (eval_metrics, mean_dice, mean_fscore,
                                   mean_iou)
from mmseg.core.evaluation.metrics import f_score


def get_confusion_matrix(pred_label, label, num_classes, ignore_index):
    """Intersection over Union
       Args:
           pred_label (np.ndarray): 2D predict map
           label (np.ndarray): label 2D label map
           num_classes (int): number of categories
           ignore_index (int): index ignore in evaluation
       """

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    n = num_classes
    inds = n * label + pred_label

    mat = np.bincount(inds, minlength=n**2).reshape(n, n)

    return mat


# This func is deprecated since it's not memory efficient
def legacy_mean_iou(results, gt_seg_maps, num_classes, ignore_index):
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_mat = np.zeros((num_classes, num_classes), dtype=np.float)
    for i in range(num_imgs):
        mat = get_confusion_matrix(
            results[i], gt_seg_maps[i], num_classes, ignore_index=ignore_index)
        total_mat += mat
    all_acc = np.diag(total_mat).sum() / total_mat.sum()
    acc = np.diag(total_mat) / total_mat.sum(axis=1)
    iou = np.diag(total_mat) / (
        total_mat.sum(axis=1) + total_mat.sum(axis=0) - np.diag(total_mat))

    return all_acc, acc, iou


# This func is deprecated since it's not memory efficient
def legacy_mean_dice(results, gt_seg_maps, num_classes, ignore_index):
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_mat = np.zeros((num_classes, num_classes), dtype=np.float)
    for i in range(num_imgs):
        mat = get_confusion_matrix(
            results[i], gt_seg_maps[i], num_classes, ignore_index=ignore_index)
        total_mat += mat
    all_acc = np.diag(total_mat).sum() / total_mat.sum()
    acc = np.diag(total_mat) / total_mat.sum(axis=1)
    dice = 2 * np.diag(total_mat) / (
        total_mat.sum(axis=1) + total_mat.sum(axis=0))

    return all_acc, acc, dice


# This func is deprecated since it's not memory efficient
def legacy_mean_fscore(results,
                       gt_seg_maps,
                       num_classes,
                       ignore_index,
                       beta=1):
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_mat = np.zeros((num_classes, num_classes), dtype=np.float)
    for i in range(num_imgs):
        mat = get_confusion_matrix(
            results[i], gt_seg_maps[i], num_classes, ignore_index=ignore_index)
        total_mat += mat
    all_acc = np.diag(total_mat).sum() / total_mat.sum()
    recall = np.diag(total_mat) / total_mat.sum(axis=1)
    precision = np.diag(total_mat) / total_mat.sum(axis=0)
    fv = np.vectorize(f_score)
    fscore = fv(precision, recall, beta=beta)

    return all_acc, recall, precision, fscore


def test_metrics():
    pred_size = (10, 30, 30)
    num_classes = 19
    ignore_index = 255
    results = np.random.randint(0, num_classes, size=pred_size)
    label = np.random.randint(0, num_classes, size=pred_size)

    # Test the availability of arg: ignore_index.
    label[:, 2, 5:10] = ignore_index

    # Test the correctness of the implementation of mIoU calculation.
    ret_metrics = eval_metrics(
        results, label, num_classes, ignore_index, metrics='mIoU')
    all_acc, acc, iou = ret_metrics['aAcc'], ret_metrics['Acc'], ret_metrics[
        'IoU']
    all_acc_l, acc_l, iou_l = legacy_mean_iou(results, label, num_classes,
                                              ignore_index)
    assert all_acc == all_acc_l
    assert np.allclose(acc, acc_l)
    assert np.allclose(iou, iou_l)
    # Test the correctness of the implementation of mDice calculation.
    ret_metrics = eval_metrics(
        results, label, num_classes, ignore_index, metrics='mDice')
    all_acc, acc, dice = ret_metrics['aAcc'], ret_metrics['Acc'], ret_metrics[
        'Dice']
    all_acc_l, acc_l, dice_l = legacy_mean_dice(results, label, num_classes,
                                                ignore_index)
    assert all_acc == all_acc_l
    assert np.allclose(acc, acc_l)
    assert np.allclose(dice, dice_l)
    # Test the correctness of the implementation of mDice calculation.
    ret_metrics = eval_metrics(
        results, label, num_classes, ignore_index, metrics='mFscore')
    all_acc, recall, precision, fscore = ret_metrics['aAcc'], ret_metrics[
        'Recall'], ret_metrics['Precision'], ret_metrics['Fscore']
    all_acc_l, recall_l, precision_l, fscore_l = legacy_mean_fscore(
        results, label, num_classes, ignore_index)
    assert all_acc == all_acc_l
    assert np.allclose(recall, recall_l)
    assert np.allclose(precision, precision_l)
    assert np.allclose(fscore, fscore_l)
    # Test the correctness of the implementation of joint calculation.
    ret_metrics = eval_metrics(
        results,
        label,
        num_classes,
        ignore_index,
        metrics=['mIoU', 'mDice', 'mFscore'])
    all_acc, acc, iou, dice, precision, recall, fscore = ret_metrics[
        'aAcc'], ret_metrics['Acc'], ret_metrics['IoU'], ret_metrics[
            'Dice'], ret_metrics['Precision'], ret_metrics[
                'Recall'], ret_metrics['Fscore']
    assert all_acc == all_acc_l
    assert np.allclose(acc, acc_l)
    assert np.allclose(iou, iou_l)
    assert np.allclose(dice, dice_l)
    assert np.allclose(precision, precision_l)
    assert np.allclose(recall, recall_l)
    assert np.allclose(fscore, fscore_l)

    # Test the correctness of calculation when arg: num_classes is larger
    # than the maximum value of input maps.
    results = np.random.randint(0, 5, size=pred_size)
    label = np.random.randint(0, 4, size=pred_size)
    ret_metrics = eval_metrics(
        results,
        label,
        num_classes,
        ignore_index=255,
        metrics='mIoU',
        nan_to_num=-1)
    all_acc, acc, iou = ret_metrics['aAcc'], ret_metrics['Acc'], ret_metrics[
        'IoU']
    assert acc[-1] == -1
    assert iou[-1] == -1

    ret_metrics = eval_metrics(
        results,
        label,
        num_classes,
        ignore_index=255,
        metrics='mDice',
        nan_to_num=-1)
    all_acc, acc, dice = ret_metrics['aAcc'], ret_metrics['Acc'], ret_metrics[
        'Dice']
    assert acc[-1] == -1
    assert dice[-1] == -1

    ret_metrics = eval_metrics(
        results,
        label,
        num_classes,
        ignore_index=255,
        metrics='mFscore',
        nan_to_num=-1)
    all_acc, precision, recall, fscore = ret_metrics['aAcc'], ret_metrics[
        'Precision'], ret_metrics['Recall'], ret_metrics['Fscore']
    assert precision[-1] == -1
    assert recall[-1] == -1
    assert fscore[-1] == -1

    ret_metrics = eval_metrics(
        results,
        label,
        num_classes,
        ignore_index=255,
        metrics=['mDice', 'mIoU', 'mFscore'],
        nan_to_num=-1)
    all_acc, acc, iou, dice, precision, recall, fscore = ret_metrics[
        'aAcc'], ret_metrics['Acc'], ret_metrics['IoU'], ret_metrics[
            'Dice'], ret_metrics['Precision'], ret_metrics[
                'Recall'], ret_metrics['Fscore']
    assert acc[-1] == -1
    assert dice[-1] == -1
    assert iou[-1] == -1
    assert precision[-1] == -1
    assert recall[-1] == -1
    assert fscore[-1] == -1

    # Test the bug which is caused by torch.histc.
    # torch.histc:  https://pytorch.org/docs/stable/generated/torch.histc.html
    # When the arg:bins is set to be same as arg:max,
    # some channels of mIoU may be nan.
    results = np.array([np.repeat(31, 59)])
    label = np.array([np.arange(59)])
    num_classes = 59
    ret_metrics = eval_metrics(
        results, label, num_classes, ignore_index=255, metrics='mIoU')
    all_acc, acc, iou = ret_metrics['aAcc'], ret_metrics['Acc'], ret_metrics[
        'IoU']
    assert not np.any(np.isnan(iou))


def test_mean_iou():
    pred_size = (10, 30, 30)
    num_classes = 19
    ignore_index = 255
    results = np.random.randint(0, num_classes, size=pred_size)
    label = np.random.randint(0, num_classes, size=pred_size)
    label[:, 2, 5:10] = ignore_index
    ret_metrics = mean_iou(results, label, num_classes, ignore_index)
    all_acc, acc, iou = ret_metrics['aAcc'], ret_metrics['Acc'], ret_metrics[
        'IoU']
    all_acc_l, acc_l, iou_l = legacy_mean_iou(results, label, num_classes,
                                              ignore_index)
    assert all_acc == all_acc_l
    assert np.allclose(acc, acc_l)
    assert np.allclose(iou, iou_l)

    results = np.random.randint(0, 5, size=pred_size)
    label = np.random.randint(0, 4, size=pred_size)
    ret_metrics = mean_iou(
        results, label, num_classes, ignore_index=255, nan_to_num=-1)
    all_acc, acc, iou = ret_metrics['aAcc'], ret_metrics['Acc'], ret_metrics[
        'IoU']
    assert acc[-1] == -1
    assert acc[-1] == -1


def test_mean_dice():
    pred_size = (10, 30, 30)
    num_classes = 19
    ignore_index = 255
    results = np.random.randint(0, num_classes, size=pred_size)
    label = np.random.randint(0, num_classes, size=pred_size)
    label[:, 2, 5:10] = ignore_index
    ret_metrics = mean_dice(results, label, num_classes, ignore_index)
    all_acc, acc, iou = ret_metrics['aAcc'], ret_metrics['Acc'], ret_metrics[
        'Dice']
    all_acc_l, acc_l, dice_l = legacy_mean_dice(results, label, num_classes,
                                                ignore_index)
    assert all_acc == all_acc_l
    assert np.allclose(acc, acc_l)
    assert np.allclose(iou, dice_l)

    results = np.random.randint(0, 5, size=pred_size)
    label = np.random.randint(0, 4, size=pred_size)
    ret_metrics = mean_dice(
        results, label, num_classes, ignore_index=255, nan_to_num=-1)
    all_acc, acc, dice = ret_metrics['aAcc'], ret_metrics['Acc'], ret_metrics[
        'Dice']
    assert acc[-1] == -1
    assert dice[-1] == -1


def test_mean_fscore():
    pred_size = (10, 30, 30)
    num_classes = 19
    ignore_index = 255
    results = np.random.randint(0, num_classes, size=pred_size)
    label = np.random.randint(0, num_classes, size=pred_size)
    label[:, 2, 5:10] = ignore_index
    ret_metrics = mean_fscore(results, label, num_classes, ignore_index)
    all_acc, recall, precision, fscore = ret_metrics['aAcc'], ret_metrics[
        'Recall'], ret_metrics['Precision'], ret_metrics['Fscore']
    all_acc_l, recall_l, precision_l, fscore_l = legacy_mean_fscore(
        results, label, num_classes, ignore_index)
    assert all_acc == all_acc_l
    assert np.allclose(recall, recall_l)
    assert np.allclose(precision, precision_l)
    assert np.allclose(fscore, fscore_l)

    ret_metrics = mean_fscore(
        results, label, num_classes, ignore_index, beta=2)
    all_acc, recall, precision, fscore = ret_metrics['aAcc'], ret_metrics[
        'Recall'], ret_metrics['Precision'], ret_metrics['Fscore']
    all_acc_l, recall_l, precision_l, fscore_l = legacy_mean_fscore(
        results, label, num_classes, ignore_index, beta=2)
    assert all_acc == all_acc_l
    assert np.allclose(recall, recall_l)
    assert np.allclose(precision, precision_l)
    assert np.allclose(fscore, fscore_l)

    results = np.random.randint(0, 5, size=pred_size)
    label = np.random.randint(0, 4, size=pred_size)
    ret_metrics = mean_fscore(
        results, label, num_classes, ignore_index=255, nan_to_num=-1)
    all_acc, recall, precision, fscore = ret_metrics['aAcc'], ret_metrics[
        'Recall'], ret_metrics['Precision'], ret_metrics['Fscore']
    assert recall[-1] == -1
    assert precision[-1] == -1
    assert fscore[-1] == -1


def test_filename_inputs():
    import cv2
    import tempfile

    def save_arr(input_arrays: list, title: str, is_image: bool, dir: str):
        filenames = []
        SUFFIX = '.png' if is_image else '.npy'
        for idx, arr in enumerate(input_arrays):
            filename = '{}/{}-{}{}'.format(dir, title, idx, SUFFIX)
            if is_image:
                cv2.imwrite(filename, arr)
            else:
                np.save(filename, arr)
            filenames.append(filename)
        return filenames

    pred_size = (10, 30, 30)
    num_classes = 19
    ignore_index = 255
    results = np.random.randint(0, num_classes, size=pred_size)
    labels = np.random.randint(0, num_classes, size=pred_size)
    labels[:, 2, 5:10] = ignore_index

    with tempfile.TemporaryDirectory() as temp_dir:

        result_files = save_arr(results, 'pred', False, temp_dir)
        label_files = save_arr(labels, 'label', True, temp_dir)

        ret_metrics = eval_metrics(
            result_files,
            label_files,
            num_classes,
            ignore_index,
            metrics='mIoU')
        all_acc, acc, iou = ret_metrics['aAcc'], ret_metrics[
            'Acc'], ret_metrics['IoU']
        all_acc_l, acc_l, iou_l = legacy_mean_iou(results, labels, num_classes,
                                                  ignore_index)
        assert all_acc == all_acc_l
        assert np.allclose(acc, acc_l)
        assert np.allclose(iou, iou_l)
