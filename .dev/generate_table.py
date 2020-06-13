import argparse
import csv
import glob
import json
import os.path as osp
from collections import OrderedDict

import mmcv

# build schedule look-up table to automatically find the final model
RESULTS_LUT = ['mIoU', 'mAcc', 'aAcc']


def get_final_iter(config):
    iter_num = config.split('_')[-2]
    assert iter_num.endswith('ki')
    return int(iter_num[:-2]) * 1000


def get_final_results(log_json_path, iter_num):
    result_dict = dict()
    with open(log_json_path, 'r') as f:
        for line in f.readlines():
            log_line = json.loads(line)
            if 'mode' not in log_line.keys():
                continue

            if log_line['mode'] == 'train' and log_line[
                    'iter'] == iter_num - 50:
                result_dict['memory'] = log_line['memory']

            if log_line['iter'] == iter_num:
                result_dict.update({
                    key: log_line[key] * 100
                    for key in RESULTS_LUT if key in log_line
                })
                return result_dict


def get_total_time(log_json_path, iter_num):

    def convert(seconds):
        hour = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60

        return f'{hour:d}:{minutes:2d}:{seconds:2d}'

    time_dict = dict()
    with open(log_json_path, 'r') as f:
        last_iter = 0
        total_sec = 0
        for line in f.readlines():
            log_line = json.loads(line)
            if 'mode' not in log_line.keys():
                continue

            if log_line['mode'] == 'train':
                cur_iter = log_line['iter']
                total_sec += (cur_iter - last_iter) * log_line['time']
                last_iter = cur_iter
        time_dict['time'] = convert(int(total_sec))

        return time_dict


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
        'out', type=str, help='output path of gathered models to be stored')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    models_root = args.root
    models_out = args.out
    config_name = args.config
    mmcv.mkdir_or_exist(models_out)

    # find all models in the root directory to be gathered
    raw_configs = list(mmcv.scandir(config_name, '.py', recursive=True))

    # filter configs that is not trained in the experiments dir
    exp_dirs = []
    for raw_config in raw_configs:
        work_dir = osp.splitext(osp.basename(raw_config))[0]
        if osp.exists(osp.join(models_root, work_dir)):
            exp_dirs.append(work_dir)
    print(f'Find {len(exp_dirs)} models to be gathered')

    # find final_ckpt and log file for trained each config
    # and parse the best performance
    model_infos = []
    for work_dir in exp_dirs:
        exp_dir = osp.join(models_root, work_dir)
        # check whether the exps is finished
        final_iter = get_final_iter(work_dir)
        final_model = 'iter_{}.pth'.format(final_iter)
        model_path = osp.join(exp_dir, final_model)

        # skip if the model is still training
        if not osp.exists(model_path):
            print(f'{model_path} not finished yet')
            continue

        # get logs
        log_json_path = glob.glob(osp.join(exp_dir, '*.log.json'))[0]
        model_performance = get_final_results(log_json_path, final_iter)

        if model_performance is None:
            continue

        head = work_dir.split('_')[0]
        backbone = work_dir.split('_')[1]
        crop_size = work_dir.split('_')[-3]
        dataset = work_dir.split('_')[-1]
        model_info = OrderedDict(
            head=head,
            backbone=backbone,
            crop_size=crop_size,
            dataset=dataset,
            iters=f'{final_iter//1000}ki')
        model_info.update(model_performance)
        model_time = get_total_time(log_json_path, final_iter)
        model_info.update(model_time)
        model_info['config'] = work_dir
        model_infos.append(model_info)

    with open(
            osp.join(models_out, 'models_table.csv'), 'w',
            newline='') as csvfile:
        writer = csv.writer(
            csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(model_infos[0].keys())
        for model_info in model_infos:
            writer.writerow(model_info.values())


if __name__ == '__main__':
    main()
