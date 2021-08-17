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
    if status_code == 403:
        flag = False
    elif status_code == 404:
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
        yml_list = [f'configs/{model_name}/{model_name}.yml']
    else:
        # check all
        yml_list = [
            f'configs/{x}/{x}.yml' for x in os.listdir('configs/')
            if x != '_base_'
        ]

    logger = get_root_logger(log_file='url_check.log', log_level=logging.ERROR)

    for yml_path in yml_list:
        # Default yaml loader unsafe.
        model_infos = yml.load(
            open(yml_path, 'r'), Loader=yml.CLoader)['Models']
        for model_info in model_infos:
            config_name = model_info['Name']
            checkpoint_url = model_info['Weights']
            status_code, flag = check_url(checkpoint_url)
            if flag:
                logger.info(
                    f'{config_name} {checkpoint_url} {status_code} valid')
            else:
                logger.error(
                    f'{config_name} {checkpoint_url} {status_code} error')


if __name__ == '__main__':
    main()
