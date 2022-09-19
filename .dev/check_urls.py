# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
from argparse import ArgumentParser

import requests
import yaml as yml

from mmseg.utils import get_root_logger


def check_url(url):
    """Check url response status.

    Args:
        url (str): url needed to check.

    Returns:
        int, bool: status code and check flag.
    """
    flag = True
    r = requests.head(url)
    status_code = r.status_code
    if status_code == 403 or status_code == 404:
        flag = False

    return status_code, flag


def parse_args():
    parser = ArgumentParser('url valid check.')
    parser.add_argument(
        '-m',
        '--model-name',
        type=str,
        help='Select the model needed to check')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_name = args.model_name

    # yml path generate.
    # If model_name is not set, script will check all of the models.
    if model_name is not None:
        yml_list = [(model_name, f'configs/{model_name}/{model_name}.yml')]
    else:
        # check all
        yml_list = [(x, f'configs/{x}/{x}.yml') for x in os.listdir('configs/')
                    if x != '_base_']

    logger = get_root_logger(log_file='url_check.log', log_level=logging.ERROR)

    for model_name, yml_path in yml_list:
        # Default yaml loader unsafe.
        model_infos = yml.load(open(yml_path), Loader=yml.CLoader)['Models']
        for model_info in model_infos:
            config_name = model_info['Name']
            checkpoint_url = model_info['Weights']
            # checkpoint url check
            status_code, flag = check_url(checkpoint_url)
            if flag:
                logger.info(f'checkpoint | {config_name} | {checkpoint_url} | '
                            f'{status_code} valid')
            else:
                logger.error(
                    f'checkpoint | {config_name} | {checkpoint_url} | '
                    f'{status_code} | error')
            # log_json check
            checkpoint_name = checkpoint_url.split('/')[-1]
            model_time = '-'.join(checkpoint_name.split('-')[:-1]).replace(
                f'{config_name}_', '')
            # two style of log_json name
            # use '_' to link model_time (will be deprecated)
            log_json_url_1 = f'https://download.openmmlab.com/mmsegmentation/v0.5/{model_name}/{config_name}/{config_name}_{model_time}.log.json'  # noqa
            status_code_1, flag_1 = check_url(log_json_url_1)
            # use '-' to link model_time
            log_json_url_2 = f'https://download.openmmlab.com/mmsegmentation/v0.5/{model_name}/{config_name}/{config_name}-{model_time}.log.json'  # noqa
            status_code_2, flag_2 = check_url(log_json_url_2)
            if flag_1 or flag_2:
                if flag_1:
                    logger.info(
                        f'log.json | {config_name} | {log_json_url_1} | '
                        f'{status_code_1} | valid')
                else:
                    logger.info(
                        f'log.json | {config_name} | {log_json_url_2} | '
                        f'{status_code_2} | valid')
            else:
                logger.error(
                    f'log.json | {config_name} | {log_json_url_1} & '
                    f'{log_json_url_2} | {status_code_1} & {status_code_2} | '
                    'error')


if __name__ == '__main__':
    main()
