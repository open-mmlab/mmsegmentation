# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import hashlib
import json
import os
import os.path as osp
import shutil

import mmcv
import torch

# build schedule look-up table to automatically find the final model
RESULTS_LUT = ['mIoU', 'mAcc', 'aAcc']


def calculate_file_sha256(file_path):
    """calculate file sha256 hash code."""
    with open(file_path, 'rb') as fp:
        sha256_cal = hashlib.sha256()
        sha256_cal.update(fp.read())
        return sha256_cal.hexdigest()


def process_checkpoint(in_file, out_file):
    checkpoint = torch.load(in_file, map_location='cpu')
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    torch.save(checkpoint, out_file)
    # The hash code calculation and rename command differ on different system
    # platform.
    sha = calculate_file_sha256(out_file)
    final_file = out_file.rstrip('.pth') + '-{}.pth'.format(sha[:8])
    os.rename(out_file, final_file)

    # Remove prefix and suffix
    final_file_name = osp.split(final_file)[1]
    final_file_name = osp.splitext(final_file_name)[0]

    return final_file_name


def get_final_iter(config):
    iter_num = config.split('_')[-2]
    assert iter_num.endswith('k')
    return int(iter_num[:-1]) * 1000


def get_final_results(log_json_path, iter_num):
    result_dict = dict()
    last_iter = 0
    with open(log_json_path, 'r') as f:
        for line in f.readlines():
            log_line = json.loads(line)
            if 'mode' not in log_line.keys():
                continue

            # When evaluation, the 'iter' of new log json is the evaluation
            # steps on single gpu.
            flag1 = ('aAcc' in log_line) or (log_line['mode'] == 'val')
            flag2 = (last_iter == iter_num - 50) or (last_iter == iter_num)
            if flag1 and flag2:
                result_dict.update({
                    key: log_line[key]
                    for key in RESULTS_LUT if key in log_line
                })
                return result_dict

            last_iter = log_line['iter']


def parse_args():
    parser = argparse.ArgumentParser(description='Gather benchmarked models')
    parser.add_argument(
        '-f', '--config-name', type=str, help='Process the selected config.')
    parser.add_argument(
        '-w',
        '--work-dir',
        default='work_dirs/',
        type=str,
        help='Ckpt storage root folder of benchmarked models to be gathered.')
    parser.add_argument(
        '-c',
        '--collect-dir',
        default='work_dirs/gather',
        type=str,
        help='Ckpt collect root folder of gathered models.')
    parser.add_argument(
        '--all', action='store_true', help='whether include .py and .log')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    work_dir = args.work_dir
    collect_dir = args.collect_dir
    selected_config_name = args.config_name
    mmcv.mkdir_or_exist(collect_dir)

    # find all models in the root directory to be gathered
    raw_configs = list(mmcv.scandir('./configs', '.py', recursive=True))

    # filter configs that is not trained in the experiments dir
    used_configs = []
    for raw_config in raw_configs:
        config_name = osp.splitext(osp.basename(raw_config))[0]
        if osp.exists(osp.join(work_dir, config_name)):
            if (selected_config_name is None
                    or selected_config_name == config_name):
                used_configs.append(raw_config)
    print(f'Find {len(used_configs)} models to be gathered')

    # find final_ckpt and log file for trained each config
    # and parse the best performance
    model_infos = []
    for used_config in used_configs:
        config_name = osp.splitext(osp.basename(used_config))[0]
        exp_dir = osp.join(work_dir, config_name)
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
                config_name=config_name,
                results=model_performance,
                iters=final_iter,
                model_time=model_time,
                log_json_path=osp.split(log_json_path)[-1]))

    # publish model for each checkpoint
    publish_model_infos = []
    for model in model_infos:
        config_name = model['config_name']
        model_publish_dir = osp.join(collect_dir, config_name)

        publish_model_path = osp.join(model_publish_dir,
                                      config_name + '_' + model['model_time'])
        trained_model_path = osp.join(work_dir, config_name,
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

        new_json_path = f'{config_name}_{model["log_json_path"]}'
        # copy log
        shutil.copy(
            osp.join(work_dir, config_name, model['log_json_path']),
            osp.join(model_publish_dir, new_json_path))

        if args.all:
            new_txt_path = new_json_path.rstrip('.json')
            shutil.copy(
                osp.join(work_dir, config_name,
                         model['log_json_path'].rstrip('.json')),
                osp.join(model_publish_dir, new_txt_path))

        if args.all:
            # copy config to guarantee reproducibility
            raw_config = osp.join('./configs', f'{config_name}.py')
            mmcv.Config.fromfile(raw_config).dump(
                osp.join(model_publish_dir, osp.basename(raw_config)))

        publish_model_infos.append(model)

    models = dict(models=publish_model_infos)
    mmcv.dump(models, osp.join(collect_dir, 'model_infos.json'), indent=4)


if __name__ == '__main__':
    main()
