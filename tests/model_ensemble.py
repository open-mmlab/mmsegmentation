import os
import sys
import numpy as np
from PIL import Image

import mmcv
import torch
import argparse
from mmcv.parallel import MMDataParallel
from mmcv.runner import (load_checkpoint, wrap_fp16_model)

from mmseg.models import build_segmentor
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmseg.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Model Ensemble Infer with logits')
    parser.add_argument('config', type=str, nargs='+', help='ensemble config files path')
    parser.add_argument('checkpoint', type=str, nargs='+', help='ensemble checkpoint files path')
    
    parser.add_argument('--out', help='the dir to save result')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='id of gpu to use')

    args = parser.parse_args()
    return args



def main(args):

    models = []
    gpu_ids = args.gpus
    configs = args.config
    ckpts = args.checkpoint

    cfg = mmcv.Config.fromfile(configs[0])
    cfg.data.test.pipeline[1].img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    cfg.data.test.pipeline[1].flip = True
    torch.backends.cudnn.benchmark = True

    # build the dataloader
    dataset = build_dataset(cfg.data.test)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        dist=False,
        shuffle=False,
    )

    for idx, (config, ckpt) in enumerate(zip(configs, ckpts)):
        cfg = mmcv.Config.fromfile(config)

        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        # fp16_cfg = cfg.get('fp16', None)
        # if fp16_cfg:
        #     wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, ckpt, map_location='cpu')
        torch.cuda.empty_cache()
        eval_kwargs = {"imgfile_prefix": args.out}    # result save dir
        tmpdir = eval_kwargs['imgfile_prefix']
        mmcv.mkdir_or_exist(tmpdir)
        model = MMDataParallel(model, device_ids=[gpu_ids[idx % len(gpu_ids)]])
        model.eval()
        models.append(model)

    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler
    for batch_indices, data in zip(loader_indices, data_loader):
        result = []
        result_logits = 0
            
        for model in models:
            with torch.no_grad():
                x, _ = scatter_kwargs(inputs=data, kwargs=None, target_gpus=model.device_ids)
                x = x[0]
                logits = model.module.aug_test_logits(**x)
                result.append(logits)

        for logit in result:
            result_logits += logit
            
        pred = result_logits.argmax(axis=1).squeeze()
        img_info = dataset.img_infos[batch_indices[0]]
        file_name = os.path.join(tmpdir, img_info['ann']['seg_map'].split('/')[-1])
        Image.fromarray(pred.astype(np.uint8)).save(file_name)
        prog_bar.update()
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
