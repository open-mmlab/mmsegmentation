import argparse
import hashlib
import json
import os
import os.path as osp
import shutil

import torch


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


def collect_log(ckpt_folder, collect_folder, config_name):
    # Extract log file. When there are multiple log file, we will throw a
    # AssertionError
    log_json_name = [
        x for x in os.listdir(ckpt_folder) if osp.splitext(x)[1] == '.json'
    ]
    assert len(
        log_json_name
    ) == 1, 'There are multiple log files, please remove useless log file.'

    log_json_name = log_json_name[0]
    log_json_path = osp.join(ckpt_folder, log_json_name)

    log_name = log_json_name.rstrip('.json')
    log_path = osp.join(ckpt_folder, log_name)

    log_json_dst_path = osp.join(collect_folder,
                                 f'{config_name}_{log_json_name}')
    log_dst_path = osp.join(collect_folder, f'{config_name}_{log_name}')

    shutil.copy(log_json_path, log_json_dst_path)
    shutil.copy(log_path, log_dst_path)

    log_time = log_name.rstrip('.log')

    print(f'[1/3] {config_name} log json collect: {log_json_path} => '
          f'{log_json_dst_path}')
    print(f'[2/3] {config_name} log collect: {log_path} => {log_dst_path}')

    return log_json_path, log_time


def collect_ckpt(log_jsons, log_time, ckpt_folder, collect_folder, config_name,
                 publish_flag, collect_mode):
    _ = log_jsons[0]
    log_jsons = log_jsons[1:]
    if collect_mode.lower() == 'last':
        src_ckpt_name = f'iter_{len(log_jsons) * 50}.pth'
        output_metric = log_jsons[-1]['IoU.referring object']
    elif collect_mode.lower() == 'best':
        max_metric = -1
        max_iter = -1
        for i, log_json in enumerate(log_jsons):
            if log_json['mode'] == 'val' or \
                    'mIoU' in log_json:
                metric = log_json['IoU.referring object']
                iter = log_json['iter']
                last_iter = log_jsons[i - 1]['iter']
                # TODO: Fix TextLoggerHook Bug
                if last_iter > iter:
                    iter = last_iter + 50
                if metric > max_metric:
                    max_iter = iter
                    max_metric = metric
        src_ckpt_name = f'iter_{max_iter}.pth'
        output_metric = max_metric

    src_ckpt_path = osp.join(ckpt_folder, src_ckpt_name)
    dst_ckpt_path = osp.join(collect_folder, f'{config_name}_{log_time}.pth')

    if not publish_flag:
        shutil.copy(src_ckpt_path, dst_ckpt_path)
        process_checkpoint(dst_ckpt_path, dst_ckpt_path)
    else:
        os.rename(src_ckpt_path, dst_ckpt_path)

    print(
        f'[3/3] {config_name} mode({collect_mode}) publish({not publish_flag})'
        f' metric({output_metric}) '
        f'ckpt collect: {src_ckpt_path} => {dst_ckpt_path}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_path',
        help='The comprehensive config path of model files'
        'which need to collect.')
    parser.add_argument('-f', '--folder', help='The model ckpts save path.')
    parser.add_argument('-c', '--collect', help='The collect folder path.')
    parser.add_argument(
        '-p',
        '--no-publish',
        action='store_true',
        help='Remove optimizer related params and calculate hash code which '
        'can verify completeness and correctness of file.')
    parser.add_argument(
        '-m',
        '--mode',
        default='best',
        type=str,
        help='The collect mode: collect best ckpt or last ckpt.')

    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config_path
    publish_flag = args.no_publish
    collect_mode = args.mode

    assert collect_mode.lower() in ['best', 'last']

    # strip dir path and file suffix
    config_name = osp.split(config_path)[1]  # strip prefix
    config_name = osp.splitext(config_name)[0]  # strip suffix

    ckpt_folder = args.folder or osp.join('work_dirs', config_name)
    collect_folder = args.collect or osp.join('collect', config_name)

    if not osp.exists(collect_folder):
        os.makedirs(collect_folder, 0o775)

    log_json_path, log_time = collect_log(ckpt_folder, collect_folder,
                                          config_name)
    log_jsons = []
    for line in open(log_json_path, 'r').readlines():
        log_jsons.append(json.loads(line))

    collect_ckpt(log_jsons, log_time, ckpt_folder, collect_folder, config_name,
                 publish_flag, collect_mode)


if __name__ == '__main__':
    main()
