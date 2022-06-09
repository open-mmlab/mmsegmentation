# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import logging
import os
import os.path as osp
import warnings
from argparse import ArgumentParser

import requests
from mmcv import Config

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.utils import get_root_logger

# ignore warnings when segmentors inference
warnings.filterwarnings('ignore')


def download_checkpoint(checkpoint_name, model_name, config_name, collect_dir):
    """Download checkpoint and check if hash code is true."""
    url = f'https://download.openmmlab.com/mmsegmentation/v0.5/{model_name}/{config_name}/{checkpoint_name}'  # noqa

    r = requests.get(url)
    assert r.status_code != 403, f'{url} Access denied.'

    with open(osp.join(collect_dir, checkpoint_name), 'wb') as code:
        code.write(r.content)

    true_hash_code = osp.splitext(checkpoint_name)[0].split('-')[1]

    # check hash code
    with open(osp.join(collect_dir, checkpoint_name), 'rb') as fp:
        sha256_cal = hashlib.sha256()
        sha256_cal.update(fp.read())
        cur_hash_code = sha256_cal.hexdigest()[:8]

    assert true_hash_code == cur_hash_code, f'{url} download failed, '
    'incomplete downloaded file or url invalid.'

    if cur_hash_code != true_hash_code:
        os.remove(osp.join(collect_dir, checkpoint_name))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint_root', help='Checkpoint file root path')
    parser.add_argument(
        '-i', '--img', default='demo/demo.png', help='Image file')
    parser.add_argument('-a', '--aug', action='store_true', help='aug test')
    parser.add_argument('-m', '--model-name', help='model name to inference')
    parser.add_argument(
        '-s', '--show', action='store_true', help='show results')
    parser.add_argument(
        '-d', '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


def inference_model(config_name, checkpoint, args, logger=None):
    cfg = Config.fromfile(config_name)
    if args.aug:
        if 'flip' in cfg.data.test.pipeline[
                1] and 'img_scale' in cfg.data.test.pipeline[1]:
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75
            ]
            cfg.data.test.pipeline[1].flip = True
        else:
            if logger is not None:
                logger.error(f'{config_name}: unable to start aug test')
            else:
                print(f'{config_name}: unable to start aug test', flush=True)

    model = init_segmentor(cfg, checkpoint, device=args.device)
    # test a single image
    result = inference_segmentor(model, args.img)

    # show the results
    if args.show:
        show_result_pyplot(model, args.img, result)
    return result


# Sample test whether the inference code is correct
def main(args):
    config = Config.fromfile(args.config)

    if not os.path.exists(args.checkpoint_root):
        os.makedirs(args.checkpoint_root, 0o775)

    # test single model
    if args.model_name:
        if args.model_name in config:
            model_infos = config[args.model_name]
            if not isinstance(model_infos, list):
                model_infos = [model_infos]
            for model_info in model_infos:
                config_name = model_info['config'].strip()
                print(f'processing: {config_name}', flush=True)
                checkpoint = osp.join(args.checkpoint_root,
                                      model_info['checkpoint'].strip())
                try:
                    # build the model from a config file and a checkpoint file
                    inference_model(config_name, checkpoint, args)
                except Exception:
                    print(f'{config_name} test failed!')
                    continue
                return
        else:
            raise RuntimeError('model name input error.')

    # test all model
    logger = get_root_logger(
        log_file='benchmark_inference_image.log', log_level=logging.ERROR)

    for model_name in config:
        model_infos = config[model_name]

        if not isinstance(model_infos, list):
            model_infos = [model_infos]
        for model_info in model_infos:
            print('processing: ', model_info['config'], flush=True)
            config_path = model_info['config'].strip()
            config_name = osp.splitext(osp.basename(config_path))[0]
            checkpoint_name = model_info['checkpoint'].strip()
            checkpoint = osp.join(args.checkpoint_root, checkpoint_name)

            # ensure checkpoint exists
            try:
                if not osp.exists(checkpoint):
                    download_checkpoint(checkpoint_name, model_name,
                                        config_name.rstrip('.py'),
                                        args.checkpoint_root)
            except Exception:
                logger.error(f'{checkpoint_name} download error')
                continue

            # test model inference with checkpoint
            try:
                # build the model from a config file and a checkpoint file
                inference_model(config_path, checkpoint, args, logger)
            except Exception as e:
                logger.error(f'{config_path} " : {repr(e)}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
