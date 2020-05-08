import argparse
import glob
import os
import os.path as osp
import shutil
import subprocess

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('src_dir', help='input experiment directory name')
    parser.add_argument('dst_dir', help='output model zoo name')
    args = parser.parse_args()
    return args


def process_checkpoint(in_file, out_file):
    checkpoint = torch.load(in_file, map_location='cpu')
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    torch.save(checkpoint, out_file)
    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    final_file = osp.splitext(out_file)[0] + '-{}.pth'.format(sha[:8])
    subprocess.Popen(['mv', out_file, final_file])


def process_log(src_dir, dst_dir, suffix='.log'):
    model_name = osp.basename(src_dir)
    log_files = glob.glob(osp.join(src_dir, '*' + suffix))
    latest_file = max(log_files, key=osp.getctime)
    shutil.copy(
        latest_file,
        osp.join(dst_dir, model_name + '_' + osp.basename(latest_file)))


def collect_model_log(src_dir, dst_dir):
    dst_dir = osp.join(dst_dir, osp.basename(src_dir))
    os.makedirs(dst_dir, exist_ok=True)
    model_name = osp.basename(src_dir) + '.pth'
    process_log(src_dir, dst_dir, suffix='.log')
    process_log(src_dir, dst_dir, suffix='.json')
    process_checkpoint(
        osp.join(src_dir, 'latest.pth'), osp.join(dst_dir, model_name))


def main():
    args = parse_args()
    collect_model_log(src_dir=args.src_dir, dst_dir=args.dst_dir)


if __name__ == '__main__':
    main()
