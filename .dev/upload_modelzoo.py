# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import oss2

ACCESS_KEY_ID = os.getenv('OSS_ACCESS_KEY_ID', None)
ACCESS_KEY_SECRET = os.getenv('OSS_ACCESS_KEY_SECRET', None)
BUCKET_NAME = 'openmmlab'
ENDPOINT = 'https://oss-accelerate.aliyuncs.com'


def parse_args():
    parser = argparse.ArgumentParser(description='Upload models to OSS')
    parser.add_argument('model_zoo', type=str, help='model_zoo input')
    parser.add_argument(
        '--dst-folder',
        type=str,
        default='mmsegmentation/v0.5',
        help='destination folder')
    return parser.parse_args()


def main():
    args = parse_args()
    model_zoo = args.model_zoo
    dst_folder = args.dst_folder
    bucket = oss2.Bucket(
        oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET), ENDPOINT, BUCKET_NAME)

    for root, dirs, files in os.walk(model_zoo):
        for file in files:
            file_path = osp.relpath(osp.join(root, file), model_zoo)
            print(f'Uploading {file_path}')

            oss2.resumable_upload(bucket, osp.join(dst_folder, file_path),
                                  osp.join(model_zoo, file_path))
            bucket.put_object_acl(
                osp.join(dst_folder, file_path), oss2.OBJECT_ACL_PUBLIC_READ)


if __name__ == '__main__':
    main()
