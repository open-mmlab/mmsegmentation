from mmengine import Config
import argparse
import os
import subprocess
from my_projects.conversion_tests.converters.conversion_dicts import(
    DATASET_CLASSES,
    DATASET_PALETTE
)
from tools.test_selection import (
    run_performance_test,
    run_confusion_analysis
)


TEST_DATASET_TEMPLATE_PATHS = {
    "HOTS"          :   "my_projects/conversion_tests/dataset_template_configs/hots_test_data.py",
    "IRL_VISION"    :   "my_projects/conversion_tests/dataset_template_configs/irl_test_data.py",
    "HOTS_CAT"      :   "my_projects/conversion_tests/dataset_template_configs/hots_cat_test_data.py",
    "IRL_VISION_CAT":   "my_projects/conversion_tests/dataset_template_configs/irl_cat_test_data.py",
    "ARID20"        :   "my_projects/conversion_tests/dataset_template_configs/arid20_cat_test_data.py",
    "ARID10"        :   "my_projects/conversion_tests/dataset_template_configs/arid10_cat_test_data.py"
}



def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir_path',
        '-mdp',
        type=str,
        default="my_projects/conversion_tests/finegrained_vs_general/selection_trained/hots_v1"
    )
    parser.add_argument(
        '--test_dataset',
        '-test_ds',
        type=str,
        default="HOTS",
        choices=[
            "HOTS", 
            "IRL_VISION",
            "HOTS_CAT", 
            "IRL_VISION_CAT",
            "ARID20",
            "ARID10",
            "ADE20K"
        ]
    )
    parser.add_argument(
        '--output_dataset',
        '-out_ds',
        type=str,
        default="HOTS",
        choices=[
            "HOTS", 
            "IRL_VISION",
            "HOTS_CAT", 
            "IRL_VISION_CAT",
            "ARID20",
            "ARID10",
            "ADE20K"
        ]
    )
    parser.add_argument(
        '--target_dataset',
        '-tar_ds',
        type=str,
        default="HOTS_CAT",
        choices=[
            "HOTS_CAT", 
            "IRL_VISION_CAT",
            "ARID20",
            "ARID10",
            "ADE20K"
        ]
    )
    
    parser.add_argument(
        '--test_results_path',
        '-trp',
        type=str,
        default="work_dirs"
    )
    parser.add_argument(
        '--gen_cfg',
        '-g_cfg',
        action='store_true'
    )
    
    parser.add_argument(
        '--cfg_save_dir',
        '-cfg_dir',
        type=str,
        default="conv_cfg"
    )
    
    args = parser.parse_args()
    return args

def conversion_test_model(
    model_path: str, 
    cfg_path: str,
    test_results_path: str,
    args
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
    
    confusion_matrix_save_path = os.path.join(
        work_dir_path, 
        "confusion_matrix"
    )
    run_confusion_matrix(
        cfg_path=cfg_path,
        prediction_result_path=prediction_result_path,
        confusion_matrix_save_path=confusion_matrix_save_path,
        args=args 
    )
    
    json_path = os.path.join(
            confusion_matrix_save_path,
            "confusion.json"
    )
    
    run_confusion_analysis(
        json_path=json_path,
        save_dir_path=confusion_matrix_save_path,
        top_n=10
    )
    
    

def run_confusion_matrix(
        cfg_path,
        prediction_result_path,
        confusion_matrix_save_path,
        args
):
    subprocess.call(
        [
            "python",
            "my_projects/conversion_tests/scripts/confusion_matrix_conversion.py",
            cfg_path,               # maybe from TEST_DATASET_TEMPLATE_PATHS
            prediction_result_path,
            confusion_matrix_save_path,
            "--test_dataset",
            args.test_dataset,
            "--output_dataset",
            args.output_dataset,
            "--target_dataset",
            args.target_dataset,
        ]
    )    

def get_checkpoint_path(model_path: str) -> str:
    return os.path.join(
        model_path,
        "weights.pth"
    )
    
def get_base_cfg_path(model_path: str) -> str:
    return os.path.join(
        model_path,
        f"{get_model_dir_name(model_path=model_path)}.py"
    )
    
def get_model_dir_name(model_path: str):
    idx_ = -2 if model_path[-1] == '/' else -1
    return model_path.split("/")[idx_]        

def gen_conversion_cfg(
    base_cfg_path: str,
    args
) -> str:    
    cfg = Config.fromfile(base_cfg_path)
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
    return cfg

def dump_conversion_cfg(
    cfg: Config,
    save_dir_path: str,
    save_file_name: str
):
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    cfg.dump(
        os.path.join(save_dir_path, save_file_name)
    )    

def gen_conversion_cfg_name(args) -> str:
    model_name = get_model_dir_name(model_path=args.model_dir_path)
    model_name = model_name.split("_")[0]
    test = args.test_dataset.lower()
    out = args.output_dataset.lower()
    target = args.target_dataset.lower()
    return f"{model_name}_{test}_{out}_{target}.py"
     
   
def gen_conversion_cfg_dir_path(args):
    return os.path.join(
        args.model_dir_path,
        args.cfg_save_dir
    )
    

def main():
    
    args = arg_parse()
    conv_cfg_dir_path = gen_conversion_cfg_dir_path(args=args)
    conv_file_name = gen_conversion_cfg_name(args=args)
    conv_cfg_path = os.path.join(conv_cfg_dir_path, conv_file_name)
    if args.gen_cfg or not os.path.exists(conv_cfg_path):
        cfg = gen_conversion_cfg(
            base_cfg_path=get_base_cfg_path(args.model_dir_path),
            args=args
        )
        dump_conversion_cfg(
            cfg=cfg,
            save_dir_path=conv_cfg_dir_path,
            save_file_name=conv_file_name
        )
    
    conversion_test_model(
        model_path=args.model_dir_path,
        cfg_path=conv_cfg_path,
        test_results_path=args.test_results_path,
        args=args
    )
    
    
        
if __name__ == '__main__':
    main()
    