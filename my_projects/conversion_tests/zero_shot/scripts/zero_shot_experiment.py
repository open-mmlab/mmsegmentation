from mmengine import Config
import argparse
import os
import subprocess
from my_projects.conversion_tests.converters.conversion_dicts import(
    DATASET_CLASSES,
    DATASET_PALETTE
)
from tools.test_selection import run_performance_test


TEST_DATASET_TEMPLATE_PATHS = {
    "HOTS" :  "my_projects/zero_shot/dataset_template_configs/hots_test_data.py",
    "IRL_VISION"   : "my_projects/zero_shot/dataset_template_configs/irl_test_data.py"
}

TEST_DATASET_RESULTS_DIR_NAME = {
    "HOTS"          :       "hots_v1",
    "IRL_VISION"    :       "irl_vision"
}

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_path",
        type=str,
        default="my_projects/zero_shot/models"
    )
    parser.add_argument(
        '--test_dataset',
        '-test_ds',
        type=str,
        default="HOTS",
        choices=["HOTS", "IRL_VISION"]
    )
    parser.add_argument(
        '--output_dataset',
        '-out_ds',
        type=str,
        default="ADE20K",
        choices=["HOTS", "IRL_VISION", "ADE20K"]
    )
    parser.add_argument(
        '--target_dataset',
        '-tar_ds',
        type=str,
        default="HOTS_CAT",
        choices=["HOTS_CAT", "IRL_VISION_CAT"]
    )
    parser.add_argument(
        '--model_name',
        '-mn',
        type=str,
        default=None,
        choices=[
            'bisenetv1', 
            'mask2former', 
            'maskformer', 
            'segformer',
            'segnext',
            None
        ]
    )
    parser.add_argument(
        '--test_results_path',
        type=str,
        default="my_projects/zero_shot/results"
    )
    parser.add_argument(
        '--gen_new_cfg',
        '-n_cfg',
        action='store_true'
    )
    
    parser.add_argument(
        '--cfg_save_dir',
        '-cfg_dir',
        type=str,
        default="temp"
    )
    
    args = parser.parse_args()
    return args

def conversion_test_model(
    model_path: str, 
    cfg_path: str,
    test_results_path: str
):
    model_name = get_model_dir_name(model_path=model_path)
    
    checkpoint_path = get_checkpoint_path(model_path=model_path)
    work_dir_path = os.path.join(
        test_results_path,
        model_name
    )
    
    prediction_result_path = os.path.join(
        test_results_path,
        model_name,
        "pred_results"
    )
    
    show_dir_path = os.path.join(
        test_results_path,
        model_name,
        "show"
    )
    run_performance_test(
        cfg_path=cfg_path, 
        checkpoint_path=checkpoint_path,
        work_dir_path=work_dir_path, 
        prediction_result_path=prediction_result_path,
        show_dir_path=show_dir_path
    )

def conversion_tests_models(args):
    for model_dir_name in os.listdir(args.models_path):
        if args.test_dataset.lower() not in model_dir_name.split("_")[-1]:
            print(f"invalid match")
            print(f"test_set: {args.test_dataset}\nmodel_dir: {model_dir_name}")
            continue
        model_path = os.path.join(
            args.models_path,
            model_dir_name
        )
         
        cfg_path = get_cfg_path(
            model_path=os.path.join(
                model_path,
                "temp"
            )
        )
        if not (
            os.path.exists(cfg_path) 
            and 
            args.test_dataset in cfg_path
            and
            args.target_dataset in cfg_path
            and 
            not args.gen_new_cfg
        ):
            
            cfg_path = gen_cfg(model_path=model_path, args=args)
        
        test_results_path = args.test_results_path
        test_dataset_dir = TEST_DATASET_RESULTS_DIR_NAME[args.test_dataset]
        if test_dataset_dir not in test_results_path:
            test_results_path = os.path.join(
                args.test_results_path,
                TEST_DATASET_RESULTS_DIR_NAME[args.test_dataset]
            ) 
        conversion_test_model(
            model_path=model_path,
            cfg_path=cfg_path,
            test_results_path=test_results_path
        )
        

def get_cfg_path(model_path: str) -> str:
    cfg_path = [
        file_name for file_name in os.listdir(model_path)
            if ".py" in file_name
    ]
    
    if cfg_path:
        return os.path.join(
            model_path,
            cfg_path[0]
        )
    return ""

def get_checkpoint_path(model_path: str) -> str:
    return os.path.join(
        model_path,
        "weights.pth"
    )

def get_model_dir_name(model_path: str):
    idx_ = -2 if model_path[-1] == '/' else -1
    return model_path.split("/")[idx_]        

def gen_cfg(model_path: str, args):
    temp_path = os.path.join(model_path, args.cfg_save_dir)
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    cfg_path = get_cfg_path(model_path=model_path)
    cfg = Config.fromfile(cfg_path)
    ds_template_cfg = Config.fromfile(
        TEST_DATASET_TEMPLATE_PATHS[args.test_dataset]
    )
    cfg.merge_from_dict(ds_template_cfg.to_dict())
    cfg.merge_from_dict(
        options=dict(
            test_evaluator = dict(
                type="CustomIoUMetricConversion",
                test_dataset=args.test_dataset,
                output_dataset=args.output_dataset,
                target_dataset=args.target_dataset
            )
        )
    )
    
    cfg.merge_from_dict(
        options=dict(
            visualizer = dict(
                name='visualizer',
                type='SegLocalVisualizerConversion',
                vis_backends=[
                    dict(type='LocalVisBackend'),
                ],
                test_dataset=args.test_dataset,
                output_dataset=args.output_dataset,
                target_dataset=args.target_dataset
            )
        )
    )
    model_name = get_model_dir_name(model_path=model_path).split("_")[0]
    dump_path = os.path.join(
        temp_path,
        f"{model_name}_{args.test_dataset}_{args.target_dataset}.py"
    )
    cfg.dump(dump_path)
    return dump_path
    

def main():
    args = arg_parse()
    
    if args.model_name is None:
        conversion_tests_models(args=args)
        
    else:
        model_name = [
            f_name for f_name in os.listdir(args.models_path)
                if args.model_name in f_name
        ][0]
        model_path = os.path.join(
            args.models_path,
            model_name
        )
        try: 
            cfg_path = get_cfg_path(
                model_path=os.path.join(
                    model_path,
                    "temp"
                )
            )
        except:
            cfg_path = gen_cfg(model_path=model_path, args=args)
        conversion_test_model(
            model_path=model_path,
            cfg_path=cfg_path,
            test_results_path=args.test_results_path
        )
        
if __name__ == '__main__':
    main()
    