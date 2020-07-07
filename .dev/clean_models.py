import argparse
import glob
import json
import os
import os.path as osp

import mmcv

# build schedule look-up table to automatically find the final model
SCHEDULES_LUT = {
    '20ki': 20000,
    '40ki': 40000,
    '60ki': 60000,
    '80ki': 80000,
    '160ki': 160000
}
RESULTS_LUT = ['mIoU', 'mAcc', 'aAcc']


def get_final_iter(config):
    iter_num = SCHEDULES_LUT[config.split('_')[-2]]
    return iter_num


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

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    models_root = args.root
    config_name = args.config

    # find all models in the root directory to be gathered
    raw_configs = list(mmcv.scandir(config_name, '.py', recursive=True))

    # filter configs that is not trained in the experiments dir
    used_configs = []
    for raw_config in raw_configs:
        work_dir = osp.splitext(osp.basename(raw_config))[0]
        if osp.exists(osp.join(models_root, work_dir)):
            used_configs.append(work_dir)
    print(f'Find {len(used_configs)} models to be gathered')

    # find final_ckpt and log file for trained each config
    # and parse the best performance
    model_infos = []
    for used_config in used_configs:
        exp_dir = osp.join(models_root, used_config)
        # check whether the exps is finished
        final_iter = get_final_iter(used_config)
        final_model = 'iter_{}.pth'.format(final_iter)
        model_path = osp.join(exp_dir, final_model)

        # skip if the model is still training
        if not osp.exists(model_path):
            print(f'{used_config} not finished yet')
            continue

        # get logs
        log_json_path = glob.glob(osp.join(exp_dir, '*.log.json'))[0]
        log_txt_path = glob.glob(osp.join(exp_dir, '*.log'))[0]
        model_performance = get_final_results(log_json_path, final_iter)

        if model_performance is None:
            print(f'{used_config} does not have performance')
            continue

        model_time = osp.split(log_txt_path)[-1].split('.')[0]
        model_infos.append(
            dict(
                config=used_config,
                results=model_performance,
                iters=final_iter,
                model_time=model_time,
                log_json_path=osp.split(log_json_path)[-1]))

    # publish model for each checkpoint
    for model in model_infos:

        model_name = osp.split(model['config'])[-1].split('.')[0]

        model_name += '_' + model['model_time']
        for checkpoints in mmcv.scandir(
                osp.join(models_root, model['config']), suffix='.pth'):
            if checkpoints.endswith(f"iter_{model['iters']}.pth"
                                    ) or checkpoints.endswith('latest.pth'):
                continue
            print('removing {}'.format(
                osp.join(models_root, model['config'], checkpoints)))
            os.remove(osp.join(models_root, model['config'], checkpoints))


if __name__ == '__main__':
    main()
