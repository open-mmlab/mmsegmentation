import torch
import numpy as np


# referring segmentation metrics (mIoU, Pr@{0.5,0.6,0.7,0.8,0.9})
def segmentation_metrics(preds, masks, device):
    iou_list = []
    for pred, mask in zip(preds, masks):
        # pred: (H, W): bool, mask: (H, W): bool
        # iou
        inter = np.logical_and(pred, mask)
        union = np.logical_or(pred, mask)
        iou = np.sum(inter) / (np.sum(union) + 1e-6)
        iou_list.append(iou)
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(device)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    head = 'Evaluation: IoU={:.2f}'.format(100. * iou.item())
    return head + temp, {'iou': iou.item(), **prec}



# eval function
def eval_dataset(test_dataset, predict_fn):
    metrics = {}
    for data in test_dataset:
        print(refer_type)
        all_preds, all_gt = [], []
        for samp in tqdm(data):

            img = samp['img']
            pred = predict_fn(img)
            
            all_preds.append(pred)
            all_gt.append(samp['mask'])

        met = segmentation_metrics(all_preds, all_gt, device)
        print(met[0])
        print()
        metrics[samp['scene_id']] = met[1]
    print(f'Average: {np.array([m["iou"] for m in metrics.values()]).mean()}')
    return metrics
