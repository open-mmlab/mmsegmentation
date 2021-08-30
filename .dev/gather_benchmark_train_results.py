import argparse
import glob
import os.path as osp

import mmcv
from gather_models import get_final_results
from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Gather benchmarked models train results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'root',
        type=str,
        help='root path of benchmarked models to be gathered')
    parser.add_argument(
        '--out',
        type=str,
        default='benchmark_train_info.json',
        help='output path of gathered metrics to be stored')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    root_path = args.root
    metrics_out = args.out

    evaluation_cfg = Config.fromfile(args.config)

    result_dict = {}
    for model_key in evaluation_cfg:
        model_infos = evaluation_cfg[model_key]
        if not isinstance(model_infos, list):
            model_infos = [model_infos]
        for model_info in model_infos:
            config = model_info['config']

            # benchmark train dir
            model_name = osp.split(osp.dirname(config))[1]
            config_name = osp.splitext(osp.basename(config))[0]
            exp_dir = osp.join(root_path, model_name, config_name)
            if not osp.exists(exp_dir):
                print(f'{config} hasn\'t {exp_dir}')
                continue

            # parse config
            cfg = mmcv.Config.fromfile(config)
            total_iters = cfg.runner.max_iters
            exp_metric = cfg.evaluation.metric
            if not isinstance(exp_metric, list):
                exp_metrics = [exp_metric]

            # determine whether total_iters ckpt exists
            ckpt_path = f'iter_{total_iters}.pth'
            if not osp.exists(osp.join(exp_dir, ckpt_path)):
                print(f'{config} hasn\'t {ckpt_path}')
                continue

            # only the last log json counts
            log_json_path = list(
                sorted(glob.glob(osp.join(exp_dir, '*.log.json'))))[-1]

            # extract metric value
            model_performance = get_final_results(log_json_path, total_iters)
            if model_performance is None:
                print(f'log file error: {log_json_path}')
                continue

            differential_results = dict()
            old_results = dict()
            new_results = dict()
            for metric_key in model_performance:
                if metric_key in ['mIoU']:
                    metric = round(model_performance[metric_key] * 100, 2)
                    old_metric = model_info['metric'][metric_key]
                    old_results[metric_key] = old_metric
                    new_results[metric_key] = metric
                    differential = metric - old_metric
                    flag = '+' if differential > 0 else '-'
                    differential_results[
                        metric_key] = f'{flag}{abs(differential):.2f}'
            result_dict[config] = dict(
                differential_results=differential_results,
                old_results=old_results,
                new_results=new_results,
            )

    # 4 save or print results
    if metrics_out:
        mmcv.dump(result_dict, metrics_out, indent=4)
    print('===================================')
    for config_name, metrics in result_dict.items():
        print(config_name, metrics)
    print('===================================')
