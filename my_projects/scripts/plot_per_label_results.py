
from matplotlib import pyplot as plt
from my_projects.scripts.generate_latex_tables import load_json_file
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data_path', 
        type=str,
        default="my_projects/test_results/arid20_cat/",
        help='directory path of experiment'
    )
    args = parser.parse_args()
    if os.path.exists(os.path.join(args.data_path, "data")):
        args.data_path = os.path.join(args.data_path, "data")
    return args
    
def save_class_hist_plot(class_hist, save_path):
    
    plt.rc('font', size=25)
    plt.figure(figsize=(30, 6))
    plt.bar(class_hist.keys(), class_hist.values(), width=0.8)
    plt.xticks(
        range(len(class_hist.keys())), class_hist.keys(), 
        rotation = 90
    )
    plt.ylabel("IoU")
    plt.ylim(48, 101)
    
    plt.savefig(save_path, bbox_inches='tight',dpi=100)
    plt.clf()

def make_class_hist(model_data_dict, metric = "IoU"):
    return {
        class_label : metric_dict[metric] 
            for class_label, metric_dict in model_data_dict.items()
    }

    

def main():
    args = parse_args()
    data_path = args.data_path
    file_name = [
        pth for pth in os.listdir(data_path) 
            if "per_label_results.json" in pth
    ][0]
    file_path = os.path.join(data_path, file_name)
    data_dict = load_json_file(json_file_path=file_path)
    for model_name, model_data in data_dict.items():
        hist = make_class_hist(model_data_dict=model_data)
        save_class_hist_plot(
            class_hist=hist,
            save_path=os.path.join(
                data_path, 
                f"{model_name}_per_label_hist.png"
            )
        )
    
if __name__ == '__main__':
    main()
