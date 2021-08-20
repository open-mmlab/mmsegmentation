import argparse
import os
import os.path as osp

# Default using 4 gpu when training
config_8gpu_list = []


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert benchmark model json to script')
    parser.add_argument(
        'txt_path', type=str, help='txt path output by benchmark_filter')
    parser.add_argument('--port', type=int, default=24727, help='dist port')
    parser.add_argument(
        '--partition',
        type=str,
        default='openmmlab',
        help='slurm partition name')
    parser.add_argument(
        '--max-keep-ckpts',
        type=int,
        default=1,
        help='The maximum checkpoints to keep')
    parser.add_argument(
        '--run', action='store_true', help='run script directly')
    parser.add_argument(
        '--out',
        type=str,
        default='.dev/benchmark_train.sh',
        help='path to save model benchmark script')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.out:
        out_suffix = args.out.split('.')[-1]
        assert args.out.endswith('.sh'), \
            f'Expected out file path suffix is .sh, but get .{out_suffix}'
    assert args.out or args.run, \
        ('Please specify at least one operation (save/run/ the '
         'script) with the argument "--out" or "--run"')

    partition = args.partition  # cluster name

    root_name = './tools'
    train_script_name = osp.join(root_name, 'slurm_train.sh')
    port = args.port
    # stdout is no output
    stdout_cfg = '>/dev/null'

    max_keep_ckpts = args.max_keep_ckpts

    commands = []
    with open(args.txt_path, 'r') as f:
        model_cfgs = f.readlines()
        for i, cfg in enumerate(model_cfgs):
            cfg = cfg.strip()
            if len(cfg) == 0:
                continue
            # print cfg name
            echo_info = f'echo \'{cfg}\' &'
            commands.append(echo_info)
            commands.append('\n')

            fname, _ = osp.splitext(osp.basename(cfg))
            # default setting
            if cfg in config_8gpu_list:
                command_info = f'GPUS=8  GPUS_PER_NODE=8  ' \
                               f'CPUS_PER_TASK=2 {train_script_name} '
            else:
                command_info = f'GPUS=4  GPUS_PER_NODE=4  ' \
                               f'CPUS_PER_TASK=2 {train_script_name} '
            command_info += f'{partition} '
            command_info += f'{fname} '
            command_info += f'{cfg} '
            if max_keep_ckpts:
                command_info += f'--options ' \
                                f'checkpoint_config.max_keep_ckpts=' \
                                f'{max_keep_ckpts} ' \
                                f'dist_params.port=' \
                                f'{port} '
            command_info += f'{stdout_cfg} &'

            commands.append(command_info)

            if i < len(model_cfgs):
                commands.append('\n')

            port += 1

        command_str = ''.join(commands)
        if args.out:
            with open(args.out, 'w') as f:
                f.write(command_str)
        if args.run:
            os.system(command_str)


if __name__ == '__main__':
    main()
