# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import json
import os
import os.path as osp
import shutil
import subprocess

import mmcv
import torch

# build schedule look-up table to automatically find the final model
RESULTS_LUT = ['mIoU', 'mAcc', 'aAcc']


def process_checkpoint(in_file, out_file):
    checkpoint = torch.load(in_file, map_location='cpu')
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    torch.save(checkpoint, out_file)
    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    final_file = out_file.rstrip('.pth') + '-{}.pth'.format(sha[:8])
    subprocess.Popen(['mv', out_file, final_file])
    return final_file


def get_final_iter(config):
    iter_num = config.split('_')[-2]
    assert iter_num.endswith('k')
    return int(iter_num[:-1]) * 1000


def get_final_results(log_json_path, iter_num):
    result_dict = dict()
    with open(log_json_path, 'r') as f:
        for line in f.readlines():
            log_line = json.loads(line)
            if 'mode' not in log_line.keys():
                continue

            if log_line['mode'] == 'train' and log_line['iter'] == iter_num:
                result_dict['memory'] = log_line['memory']

            if log_line['iter'] == iter_num:
                result_dict.update({
                    key: log_line[key]
                    for key in RESULTS_LUT if key in log_line
                })
                return result_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Gather benchmarked models')
    parser.add_argument(
        'root',
        type=str,
        help='root path of benchmarked models to be gathered')
    parser.add_argument(
        'config',
        type=str,
        help='root path of benchmarked configs to be gathered')
    parser.add_argument(
        'out_dir',
        type=str,
        help='output path of gathered models to be stored')
    parser.add_argument('out_file', type=str, help='the output json file name')
    parser.add_argument(
        '--filter', type=str, nargs='+', default=[], help='config filter')
    parser.add_argument(
        '--all', action='store_true', help='whether include .py and .log')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    models_root = args.root
    models_out = args.out_dir
    config_name = args.config
    mmcv.mkdir_or_exist(models_out)

    # find all models in the root directory to be gathered
    raw_configs = list(mmcv.scandir(config_name, '.py', recursive=True))

    # filter configs that is not trained in the experiments dir
    used_configs = []
    for raw_config in raw_configs:
        work_dir = osp.splitext(osp.basename(raw_config))[0]
        if osp.exists(osp.join(models_root, work_dir)):
            used_configs.append((work_dir, raw_config))
    print(f'Find {len(used_configs)} models to be gathered')

    # find final_ckpt and log file for trained each config
    # and parse the best performance
    model_infos = []
    for used_config, raw_config in used_configs:
        bypass = True
        for p in args.filter:
            if p in used_config:
                bypass = False
                break
        if bypass:
            continue
        exp_dir = osp.join(models_root, used_config)
        # check whether the exps is finished
        final_iter = get_final_iter(used_config)
        final_model = 'iter_{}.pth'.format(final_iter)
        model_path = osp.join(exp_dir, final_model)

        # skip if the model is still training
        if not osp.exists(model_path):
            print(f'{used_config} train not finished yet')
            continue

        # get logs
        log_json_paths = glob.glob(osp.join(exp_dir, '*.log.json'))
        log_json_path = log_json_paths[0]
        model_performance = None
        for idx, _log_json_path in enumerate(log_json_paths):
            model_performance = get_final_results(_log_json_path, final_iter)
            if model_performance is not None:
                log_json_path = _log_json_path
                break

        if model_performance is None:
            print(f'{used_config} model_performance is None')
            continue

        model_time = osp.split(log_json_path)[-1].split('.')[0]
        model_infos.append(
            dict(
                config=used_config,
                raw_config=raw_config,
                results=model_performance,
                iters=final_iter,
                model_time=model_time,
                log_json_path=osp.split(log_json_path)[-1]))

    # publish model for each checkpoint
    publish_model_infos = []
    for model in model_infos:
        model_publish_dir = osp.join(models_out,
                                     model['raw_config'].rstrip('.py'))
        model_name = osp.split(model['config'])[-1].split('.')[0]

        publish_model_path = osp.join(model_publish_dir,
                                      model_name + '_' + model['model_time'])
        trained_model_path = osp.join(models_root, model['config'],
                                      'iter_{}.pth'.format(model['iters']))
        if osp.exists(model_publish_dir):
            for file in os.listdir(model_publish_dir):
                if file.endswith('.pth'):
                    print(f'model {file} found')
                    model['model_path'] = osp.abspath(
                        osp.join(model_publish_dir, file))
                    break
            if 'model_path' not in model:
                print(f'dir {model_publish_dir} exists, no model found')

        else:
            mmcv.mkdir_or_exist(model_publish_dir)

            # convert model
            final_model_path = process_checkpoint(trained_model_path,
                                                  publish_model_path)
            model['model_path'] = final_model_path

        new_json_path = f'{model_name}-{model["log_json_path"]}'
        # copy log
        shutil.copy(
            osp.join(models_root, model['config'], model['log_json_path']),
            osp.join(model_publish_dir, new_json_path))
        if args.all:
            new_txt_path = new_json_path.rstrip('.json')
            shutil.copy(
                osp.join(models_root, model['config'],
                         model['log_json_path'].rstrip('.json')),
                osp.join(model_publish_dir, new_txt_path))

        if args.all:
            # copy config to guarantee reproducibility
            raw_config = osp.join(config_name, model['raw_config'])
            mmcv.Config.fromfile(raw_config).dump(
                osp.join(model_publish_dir, osp.basename(raw_config)))

        publish_model_infos.append(model)

    models = dict(models=publish_model_infos)
    mmcv.dump(models, osp.join(models_out, args.out_file))


if __name__ == '__main__':
    main()
