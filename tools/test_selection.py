import os
import subprocess
import argparse


def run_performance_test(
    cfg_path, checkpoint_path, 
    work_dir_path, prediction_result_path, 
    show_dir_path
):
    call_list = ["python", "tools/test.py"]
    call_list.append(cfg_path)
    call_list.append(checkpoint_path)
    
    # specify workdir arg
    call_list.append("--work-dir")
    call_list.append(work_dir_path)

    # # out args
    call_list.append("--out")
    call_list.append(prediction_result_path)
        
    # # activate show
    # call_list.append("--show")
    # Add show dir arg
    call_list.append("--show-dir")
    call_list.append(show_dir_path)
    
    subprocess.call(call_list)

def run_benchmark(
    cfg_path, checkpoint_path, 
    work_dir_path, repeat_times = 1
):
    call_list = ["python", "tools/analysis_tools/benchmark.py"]
    call_list.append(cfg_path)
    call_list.append(checkpoint_path)
    
    # specify workdir arg
    call_list.append("--work-dir")
    call_list.append(work_dir_path)
    
    call_list.append("--repeat-times")
    call_list.append(str(repeat_times))
    
    subprocess.call(call_list)


def run_clutter_test(
    cfg_path, checkpoint_path, 
    work_dir_path
):
    # run clutter experiment
    call_list = ["python", "tools/clutter_test.py"]
    call_list.append(cfg_path)
    call_list.append(checkpoint_path)
    
    # specify workdir arg
    call_list.append("--work-dir")
    call_list.append(work_dir_path)
    subprocess.call(call_list)
    
# def run_scene_test(
#     cfg_path, checkpoint_path, 
#     work_dir_path
# ):
#     # run scene experiment
#     call_list = ["python", "my_projects/ablation_tests/scripts/scene_test.py"]
#     call_list.append(cfg_path)
#     call_list.append(checkpoint_path)
    
#     # specify workdir arg
#     call_list.append("--work-dir")
#     call_list.append(work_dir_path)
#     subprocess.call(call_list)   

def run_confusion_matrix(
    cfg_path, prediction_result_path,
    confusion_matrix_save_path
):
    call_list = ["python", "tools/analysis_tools/confusion_matrix.py"]
    
    call_list.append(cfg_path)
    
    call_list.append(prediction_result_path)
    
    call_list.append(confusion_matrix_save_path)
    
    subprocess.call(call_list)

def run_confusion_analysis(
    json_path, save_dir_path, top_n = 10
):
    call_list = ["python"]
    call_list.append("tools/analysis_tools/analyze_confusion_matrix_data.py")

    call_list.append(json_path)
    
    call_list.append("--top_n")
    call_list.append(str(top_n))
    
    call_list.append("--ignore_background")
    
    call_list.append("--save_dir_path")
    call_list.append(save_dir_path)
    
    subprocess.call(call_list)
    
    
def run_flops(
    cfg_path, work_dir_path, shape = [3, 512, 512]
):
    call_list = ["python", "tools/analysis_tools/get_flops.py"]
    
    call_list.append(cfg_path)
    
    call_list.append("--shape")
    
    for dim in shape[1:]:
        call_list.append(str(dim))
    
    call_list.append("--work-dir")
    call_list.append(work_dir_path)
    subprocess.call(call_list)

# results path is the parent folder of the modeldirs
def collect_and_organize_all_data(
    results_path
):
    # organize and plot clutterdata per model
    for model_dir in os.listdir(results_path):
        if model_dir == "data":
            continue
        model_path = os.path.join(results_path, model_dir)
        if not os.path.isdir(model_path):
            continue
        clutter_path = os.path.join(model_path, "clutter")
        if not os.path.exists(clutter_path):
            continue
        model_name = model_dir.split("_")[0]
        call_list = [
            "python", 
            "my_projects/scripts/organise_clutter_results.py",
            clutter_path
        ]
        subprocess.call(call_list)
    
    
    # plot clutterdata global and per model 
    work_dir_name = results_path.split("/")[-2 if results_path[-1] == '/' else -1]
    data_path = os.path.join(results_path, "data")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    call_list = [
        "python",
        "my_projects/scripts/plot_clutter_data.py",
        results_path,
        data_path,
        "--global_plot",
        '--per_model'
    ]
    subprocess.call(call_list)
    
    # generate jsons
    call_list = [
        "python",
        "my_projects/scripts/generate_json_all_test_results.py",
        "-dsp",
        results_path,
        "-si"
    ]
    subprocess.call(call_list)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'models_path',
        type=str
    )
    parser.add_argument(
        'test_results_path',
        type=str
    )
    args = parser.parse_args()
    return args
   
def main():
    # selection_path = "my_projects/best_models/selection_trained/arid20_cat"
    # test_results_path = "my_projects/test_results/arid20_cat"
    args = parse_args()
    selection_path = args.models_path
    test_results_path = args.test_results_path
    for model_name in os.listdir(selection_path):
        if model_name == "data":
            continue
        
        cfg_path = os.path.join(
            selection_path, 
            model_name, 
            f"{model_name}.py"
        )
    
        
        checkpoint_path = os.path.join(
            selection_path,
            model_name,
            "weights.pth"
        )
        
        
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
        
        run_clutter_test(
            cfg_path=cfg_path,
            checkpoint_path=checkpoint_path,
            work_dir_path=work_dir_path
        )
        # run_scene_test( 
        #     cfg_path=cfg_path,
        #     checkpoint_path=checkpoint_path,
        #     work_dir_path=work_dir_path
        # )
        bench_work_dir_path = os.path.join(work_dir_path, "benchmark")
        
        run_benchmark(
            cfg_path=cfg_path,
            checkpoint_path=checkpoint_path,
            work_dir_path=bench_work_dir_path,
            repeat_times=10
        )

        confusion_matrix_save_path = os.path.join(
            work_dir_path, 
            "confusion_matrix"
        )
        
        run_confusion_matrix(
            cfg_path=cfg_path,
            prediction_result_path=prediction_result_path,
            confusion_matrix_save_path=confusion_matrix_save_path
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
        
        save_flops_file_path = os.path.join(
            work_dir_path,
            "flops"
        )
        
        run_flops(
            cfg_path=cfg_path,
            work_dir_path=save_flops_file_path,
            shape=[3, 512, 512]
        )
    collect_and_organize_all_data(results_path=test_results_path)    
        
        
        
if __name__ == '__main__':
    main()


