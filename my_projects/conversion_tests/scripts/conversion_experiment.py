import subprocess
import os
from tools.test_selection import collect_and_organize_all_data


DATASET_SOURCE_PATH = {
    "HOTS"          :       "my_projects/conversion_tests/selection_trained/hots_v1",
    "IRL_VISION"    :       "my_projects/conversion_tests/selection_trained/irl_vision",
    "HOTS_CAT"      :       "my_projects/conversion_tests/selection_trained/hots_v1_cat",
    "IRL_VISION_CAT":       "my_projects/conversion_tests/selection_trained/irl_vision_cat",
    "ARID20"        :       "my_projects/conversion_tests/selection_trained/arid20_cat",
    "ARID10"        :       "my_projects/conversion_tests/selection_trained/arid10_cat",
    "ADE20K"        :       "my_projects/conversion_tests/zero_shot/models",
    "SODHOTS"       :       "my_projects/conversion_tests/selection_trained/sodhots-c"
}

# (test_dataset, output_dataset, target_dataset)
# with:
#  - test_dataset : name of test set used, images in test loader
#  - output_dataset : dataset the model is trained for and the corresponding 
#                     labels it outputs
#  - target_dataset : how the labels of both test set and output set should 
#                     be converted: tot the target set
DATASET_SETUPS = [
    # ("HOTS_CAT", "HOTS", "HOTS_CAT"),
    # ("IRL_VISION_CAT", "IRL_VISION", "IRL_VISION_CAT"),
    # ("IRL_VISION_CAT", "HOTS_CAT", "IRL_VISION_CAT"),
    # ("HOTS_CAT", "IRL_VISION_CAT", "HOTS_CAT")
    # ("HOTS_CAT", "IRL_VISION", "HOTS_CAT"),
    # ("IRL_VISION_CAT", "HOTS", "IRL_VISION_CAT")
    # ("ARID20", "ARID10", "ARID20"),
    ("HOTS_CAT", "ADE20K", "ADE20K")
]

def run_conversion_test(
    model_dir_path: str,
    test_dataset: str,
    output_dataset: str,
    target_dataset: str,
    test_results_path: str,
    gen_cfg: bool = True,
    cfg_save_dir: str = "conv_cfg"
):
    subprocess.call(
        [
            "python",
            "my_projects/conversion_tests/scripts/conversion_test.py",
            "--model_dir_path",
            model_dir_path,
            "--test_dataset",
            test_dataset,
            "--output_dataset",
            output_dataset,
            "--target_dataset",
            target_dataset,
            "--test_results_path",
            test_results_path,
            "--gen_cfg",
            "--cfg_save_dir",
            cfg_save_dir  
        ]
    )
    
    
    

def main():
    results_dir_path = "my_projects/conversion_tests/test_results"
    for (test_dataset, output_dataset, target_dataset) in DATASET_SETUPS:
        dataset_src_path = DATASET_SOURCE_PATH[output_dataset]
        test_results_path = os.path.join(
            results_dir_path,
            f"{output_dataset.lower()}2{target_dataset.lower()}"
        )
        for model_dir in os.listdir(dataset_src_path):
            run_conversion_test(
                model_dir_path=os.path.join(
                    dataset_src_path,
                    model_dir
                ),
                test_dataset=test_dataset,
                output_dataset=output_dataset,
                target_dataset=target_dataset,
                test_results_path=test_results_path,
                gen_cfg=True,
                cfg_save_dir="conv_cfg"
            )
        collect_and_organize_all_data(
            results_path=test_results_path
        )

if __name__ == '__main__':
    main()