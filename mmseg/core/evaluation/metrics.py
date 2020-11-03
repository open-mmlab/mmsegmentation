import numpy as np


def intersect_and_union(pred_label, label, num_classes, ignore_index):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray): Prediction segmentation map
        label (ndarray): Ground truth segmentation map
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes
         ndarray: The union of prediction and ground truth histogram on all
             classes
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results, gt_seg_maps, num_classes, ignore_index):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes
         ndarray: The union of prediction and ground truth histogram on all
             classes
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=np.float)
    total_area_union = np.zeros((num_classes, ), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes, ), dtype=np.float)
    total_area_label = np.zeros((num_classes, ), dtype=np.float)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index=ignore_index)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, \
        total_area_pred_label, total_area_label


def mean_iou(results, gt_seg_maps, num_classes, ignore_index, nan_to_num=None):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, )
         ndarray: Per category IoU, shape (num_classes, )
    """

    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(results, gt_seg_maps,
                                                     num_classes,
                                                     ignore_index=ignore_index)
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / total_area_label
    iou = total_area_intersect / total_area_union
    if nan_to_num is not None:
        return all_acc, np.nan_to_num(acc, nan=nan_to_num), \
            np.nan_to_num(iou, nan=nan_to_num)
    return all_acc, acc, iou


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, )
         ndarray: Per category dice, shape (num_classes, )
    """

    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(results, gt_seg_maps,
                                                     num_classes,
                                                     ignore_index=ignore_index)
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / total_area_label
    dice = 2 * total_area_intersect / (
        total_area_pred_label + total_area_label)
    if nan_to_num is not None:
        return all_acc, np.nan_to_num(acc, nan=nan_to_num), \
            np.nan_to_num(dice, nan=nan_to_num)
    return all_acc, acc, dice


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metric='mIoU',
                 nan_to_num=None):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        num_classes (int): Number of categories
        ignore_index (int): Index that will be ignored in evaluation.
        metric (str): Metrics to be evaluated, 'mIoU' or 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, )
         ndarray: Per category evalution metrics, shape (num_classes, )
    """

    allowed_metrics = {'mIoU': mean_iou, 'mDice': mean_dice}
    if (not isinstance(metric, str)) or (metric not in allowed_metrics):
        raise KeyError('metric {} is not supported'.format(metric))
    all_acc, acc, eval_metric = allowed_metrics[metric](results, gt_seg_maps,
                                                        num_classes,
                                                        ignore_index,
                                                        nan_to_num)
    return all_acc, acc, eval_metric
