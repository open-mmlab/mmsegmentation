from class_names import get_classes, get_palette
from eval_hooks import DistEvalHook, EvalHook
from metrics import eval_metrics, mean_dice, mean_iou
from pytorch_metrics import torch_eval_metrics, torch_mean_dice, torch_mean_iou

__all__ = [
    'EvalHook', 'DistEvalHook', 'mean_dice', 'mean_iou', 'eval_metrics',
    'get_classes', 'get_palette'
]

if __name__ == '__main__':
    import numpy as np
    import time
    pred_size = (100, 512, 1024)
    num_classes = 19
    ignore_index = 255
    results = np.random.randint(0, num_classes, size=pred_size)
    label = np.random.randint(0, num_classes, size=pred_size)
    label[:, 2, 5:10] = ignore_index
    start = time.time()
    all_acc1, acc1, iou1 = eval_metrics(
        results, label, num_classes, ignore_index, metrics='mIoU')
    print(time.time() - start)

    start = time.time()
    all_acc2, acc2, iou2 = torch_eval_metrics(
        results, label, num_classes, ignore_index, metrics='mIoU')
    print(time.time() - start)
    assert all_acc1 == all_acc2
    assert np.allclose(acc1, acc2)
    assert np.allclose(iou1, iou2)

    start = time.time()
    all_acc3, acc3, iou1 = mean_iou(results, label, num_classes, ignore_index)
    print(time.time() - start)

    start = time.time()
    all_acc4, acc4, iou2 = torch_mean_iou(results, label, num_classes,
                                          ignore_index)
    print(time.time() - start)

    assert all_acc3 == all_acc4
    assert np.allclose(acc3, acc4)
    assert np.allclose(iou1, iou2)

    start = time.time()
    all_acc5, acc5, dice1 = mean_dice(results, label, num_classes,
                                      ignore_index)
    print(time.time() - start)

    start = time.time()
    all_acc6, acc6, dice2 = torch_mean_dice(results, label, num_classes,
                                            ignore_index)
    print(time.time() - start)
    assert all_acc5 == all_acc6
    assert np.allclose(acc5, acc6)
    assert np.allclose(dice1, dice2)
