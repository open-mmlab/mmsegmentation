# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

# Default using 4 gpu when training
config_8gpu_list = [
    'configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py',  # noqa
    'configs/vit/upernet_vit-b16_ln_mln_512x512_160k_ade20k.py',
    'configs/vit/upernet_deit-s16_ln_mln_512x512_160k_ade20k.py',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert benchmark model json to script')
    parser.add_argument(
        'txt_path', type=str, help='txt path output by benchmark_filter')
    parser.add_argument('--port', type=int, default=24727, help='dist port')
    parser.add_argument(
        '--out',
        type=str,
        default='.dev/benchmark_train.sh',
        help='path to save model benchmark script')

    args = parser.parse_args()
    return args


def create_train_bash_info(commands, config, script_name, partition, port):
    cfg = config.strip()

    # print cfg name
    echo_info = f'echo \'{cfg}\' &'
    commands.append(echo_info)
    commands.append('\n')

    _, model_name = osp.split(osp.dirname(cfg))
    config_name, _ = osp.splitext(osp.basename(cfg))
    # default setting
    if cfg in config_8gpu_list:
        command_info = f'GPUS=8  GPUS_PER_NODE=8  ' \
                        f'CPUS_PER_TASK=2 {script_name} '
    else:
        command_info = f'GPUS=4  GPUS_PER_NODE=4  ' \
                        f'CPUS_PER_TASK=2 {script_name} '
    command_info += f'{partition} '
    command_info += f'{config_name} '
    command_info += f'{cfg} '
    command_info += f'--options ' \
                    f'checkpoint_config.max_keep_ckpts=1 ' \
                    f'dist_params.port={port} '
    command_info += f'--work-dir work_dirs/{model_name}/{config_name} '
    # Let the script shut up
    command_info += '>/dev/null &'

    commands.append(command_info)
    commands.append('\n')


def main():
    args = parse_args()
    if args.out:
        out_suffix = args.out.split('.')[-1]
        assert args.out.endswith('.sh'), \
            f'Expected out file path suffix is .sh, but get .{out_suffix}'

    root_name = './tools'
    script_name = osp.join(root_name, 'slurm_train.sh')
    port = args.port
    partition_name = 'PARTITION=$1'

    commands = []
    commands.append(partition_name)
    commands.append('\n')
    commands.append('\n')

    with open(args.txt_path, 'r') as f:
        model_cfgs = f.readlines()
        for i, cfg in enumerate(model_cfgs):
            create_train_bash_info(commands, cfg, script_name, '$PARTITION',
                                   port)
            port += 1

        command_str = ''.join(commands)
        if args.out:
            with open(args.out, 'w') as f:
                f.write(command_str)


if __name__ == '__main__':
    main()
