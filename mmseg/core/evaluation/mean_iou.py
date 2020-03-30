import numpy as np


def intersect_over_union(pred_label, label, num_classes, ignore_index):
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


def mean_iou(results, gt_seg_maps, num_classes, ignore_index):
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_mat = np.zeros((num_classes, num_classes), dtype=np.float)
    for i in range(num_imgs):
        mat = intersect_over_union(
            results[i], gt_seg_maps[i], num_classes, ignore_index=ignore_index)
        total_mat += mat
    all_acc = np.diag(total_mat).sum() / total_mat.sum()
    acc = np.diag(total_mat) / total_mat.sum(axis=1)
    iou = np.diag(total_mat) / (
        total_mat.sum(axis=1) + total_mat.sum(axis=0) - np.diag(total_mat))

    return all_acc, acc, iou
