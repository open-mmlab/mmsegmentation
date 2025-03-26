import subprocess
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--models_path', 
        type=str,
        default="my_projects/best_models/selection_trained/arid20_cat",
        help='path of models'
    )
    parser.add_argument(
        '--results_path',
        type=str,
        help='path to store results',
        default='my_projects/ablation_tests/test_results'
    )
    args = parser.parse_args()
    

    return args

def main():
    args = parse_args()
    for model_name in os.listdir(args.models_path):
        cfg_path = os.path.join(
            args.models_path,
            model_name,
            f"{model_name}.py"
        )
        weights_path = os.path.join(
            args.models_path,
            model_name,
            "weights.pth"
        )
        work_dir_path = os.path.join(
            args.results_path,
            model_name
        )
        subprocess.call(
            [
                "python",
                "my_projects/ablation_tests/scripts/scene_test.py",
                cfg_path,
                weights_path,
                "--work-dir",
                work_dir_path
            ]
        )
        
        
    
    
if __name__ == '__main__':
    main()