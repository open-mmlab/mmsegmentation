import numpy as np
import sklearn.metrics as sk
from mmcv.utils import print_log
from ood_metrics import fpr_at_95_tpr
RECALL_LEVEL_DEFAULT = 0.95

np.seterr(invalid='ignore')


def expected_calibration_error(pred, label, num_bins=5):
    pred_y = np.argmax(pred, axis=-1)
    correct = (pred_y == label).astype(np.float32)
    prob_y = np.max(pred, axis=-1)

    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins=b, right=True)

    o = 0
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))

    return o / pred.shape[0]


def get_measures(_pos, _neg):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_at_95_tpr(examples, labels)
    ece = expected_calibration_error(examples, labels)

    return auroc, aupr, fpr, ece


def print_measures(auroc, aupr, fpr, ece, logger=None, text="max_softmax"):
    print_log(f'OOD metrics computed using: {text}')
    print_log('FPR{:d}: {:.2f}'.format(int(100 * RECALL_LEVEL_DEFAULT), 100 * fpr), logger)
    print_log('AUROC: {:.2f}'.format(100 * auroc), logger)
    print_log('AUPR: {:.2f}'.format(100 * aupr), logger)
    print_log('ECE: {:.2f}'.format(100 * ece), logger)


def print_measures_with_std(aurocs, auprs, fprs, eces, logger=None, text="max_softmax"):
    print_log(f'OOD metrics computed using: {text}')
    print_log('FPR{:d}: {:.2f} +/- {:.2f}'.format(int(100 * RECALL_LEVEL_DEFAULT), 100 * np.mean(fprs), 100 * np.std(fprs)), logger)
    print_log('AUROC: {:.2f} +/- {:.2f}'.format(100 * np.mean(aurocs), 100 * np.std(aurocs)), logger)
    print_log('AUPR: {:.2f} +/- {:.2f}'.format(100 * np.mean(auprs), 100 * np.std(auprs)), logger)
    print_log('ECE: {:.2f} +/- {:.2f}'.format(100 * np.mean(eces), 100 * np.std(eces)), logger)
