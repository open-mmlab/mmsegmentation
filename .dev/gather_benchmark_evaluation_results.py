# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os.path as osp

import mmcv
from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Gather benchmarked model evaluation results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'root',
        type=str,
        help='root path of benchmarked models to be gathered')
    parser.add_argument(
        '--out',
        type=str,
        default='benchmark_evaluation_info.json',
        help='output path of gathered metrics and compared '
        'results to be stored')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    root_path = args.root
    metrics_out = args.out
    result_dict = {}

    cfg = Config.fromfile(args.config)

    for model_key in cfg:
        model_infos = cfg[model_key]
        if not isinstance(model_infos, list):
            model_infos = [model_infos]
        for model_info in model_infos:
            previous_metrics = model_info['metric']
            config = model_info['config'].strip()
            fname, _ = osp.splitext(osp.basename(config))

            # Load benchmark evaluation json
            metric_json_dir = osp.join(root_path, fname)
            if not osp.exists(metric_json_dir):
                print(f'{metric_json_dir} not existed.')
                continue

            json_list = glob.glob(osp.join(metric_json_dir, '*.json'))
            if len(json_list) == 0:
                print(f'There is no eval json in {metric_json_dir}.')
                continue

            log_json_path = list(sorted(json_list))[-1]
            metric = mmcv.load(log_json_path)
            if config not in metric.get('config', {}):
                print(f'{config} not included in {log_json_path}')
                continue

            # Compare between new benchmark results and previous metrics
            differential_results = {}
            new_metrics = {}
            for record_metric_key in previous_metrics:
                if record_metric_key not in metric['metric']:
                    raise KeyError('record_metric_key not exist, please '
                                   'check your config')
                old_metric = previous_metrics[record_metric_key]
                new_metric = round(metric['metric'][record_metric_key] * 100,
                                   2)

                differential = new_metric - old_metric
                flag = '+' if differential > 0 else '-'
                differential_results[
                    record_metric_key] = f'{flag}{abs(differential):.2f}'
                new_metrics[record_metric_key] = new_metric

            result_dict[config] = dict(
                differential=differential_results,
                previous=previous_metrics,
                new=new_metrics)

    if metrics_out:
        mmcv.dump(result_dict, metrics_out, indent=4)
    print('===================================')
    for config_name, metrics in result_dict.items():
        print(config_name, metrics)
    print('===================================')
