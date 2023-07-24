# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from mmseg.utils import push_to_hf_hub, has_hf_hub


def parse_args():
    parser = argparse.ArgumentParser(
        description='Upload model to Hugging Face Hub')

    parser.add_argument('model', help='model name in metafiles')
    parser.add_argument(
        '--repo-id', default=None, type=str, help='repo-id for this model')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert has_hf_hub(True)
    repo_id = args.repo_id if args.repo_id is not None else args.model
    push_to_hf_hub(args.model, repo_id=repo_id)
