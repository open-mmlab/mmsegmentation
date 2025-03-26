import subprocess
import os
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'selection_path',
        type=str
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config_dir = args.selection_path
    for cfg_name in os.listdir(config_dir):
        # if "mask" not in cfg_name:
        #     continue
        torch.cuda.empty_cache()
        
        cfg_path = os.path.join(config_dir, cfg_name)
        print(f"cfg: {cfg_name}")
        call_list = [
            "python3",
            "tools/train.py",
            cfg_path 
        ]
        subprocess.call(call_list)
        



if __name__ == '__main__':
    main()


