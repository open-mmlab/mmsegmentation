
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


# ONLY WORKS FOR ARID20_CAT

# load cfg
# change test_path testloader to n in [0-19] 
# run test for each n
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    args = parser.parse_args()
    

    return args



def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    work_dir = cfg.work_dir
    
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    
    # TODO implement global test
    # cfg.work_dir = os.path.join(work_dir, "global")
    # runner = Runner.from_cfg(cfg)
    # # start testing
    # runner.test()
    
    test_dataloader = cfg.test_dataloader
    
    dataset_root = test_dataloader["dataset"]["data_root"]
    scene_dir_path = os.path.join(dataset_root, "img_dir", "scene")
    
    
    
    if not os.path.exists(scene_dir_path):
        print(f"path {scene_dir_path} does not exists. scene test failed")
        return
    
    for scene in os.listdir(scene_dir_path):
        
        
        test_dataloader["dataset"]["data_prefix"] = dict(
            img_path=os.path.join("img_dir", "scene", scene),
            seg_map_path=os.path.join("ann_dir", "scene", scene)
        )
        cfg.test_dataloader = test_dataloader
        cfg.work_dir = os.path.join(work_dir, scene)

        runner = Runner.from_cfg(cfg)

        # start testing
        runner.test()


if __name__ == '__main__':
    main()
