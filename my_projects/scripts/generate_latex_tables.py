from copy import deepcopy
import json

# TODO change std out to filepath

LABEL_TABLE_TEMPLATE = """
    \\begin{table}[ht]
    \centering
    \caption{CAPTION}
    \\begin{adjustbox}{width=\columnwidth,center}
    \\begin{tabular}{@{}l|rrrrrr@{}}
    \\toprule
    #
    \\bottomrule
    \end{tabular}
    \end{adjustbox}
    \label{LABEL}
    \end{table} 
    """
LABEL_MODEL_TABLE_TEMPLATE = """
    \\begin{table}[ht]
    \centering
    \caption{CAPTION}
    \\begin{tabular}{@{}l|ccccc@{}}
    \\toprule
    Model & MODEL1 & MODEL2 & MODEL3 & MODEL4 & MODEL5 \\\\
    \\midrule
    Label & \\multicolumn{5}{c}{IoU} \\\\
    \\midrule
    #
    \\bottomrule
    \end{tabular}
    \label{LABEL}
    \end{table} 
    """
def get_label_tab_template(model_name, dataset_name):
    template = deepcopy(LABEL_TABLE_TEMPLATE)
    caption=f"Performance Metrics per Class {model_name} ({dataset_name})"
    label=f"tab:per_class_{model_name}_{dataset_name.lower().replace(' ', '_')}"
    template = template.replace("CAPTION", caption)
    template = template.replace("LABEL", label)
    return template

def get_label_model_template(dataset_name, model_names):
    template = deepcopy(LABEL_MODEL_TABLE_TEMPLATE)
    caption=f"Performance Metrics per Class {dataset_name}"
    label=f"tab:per_class_{dataset_name.lower().replace(' ', '_')}"
    template = template.replace("CAPTION", caption)
    template = template.replace("LABEL", label)
    for model_idx, model_name in enumerate(model_names):
        template = template.replace(f"MODEL{model_idx + 1}", model_name)
    return template

def per_label_table(
    label_results_dict, 
    model_name,
    dataset_name,
    col_names = [
        "Class", "Pr\\textsubscript{50}", 
        "Pr\\textsubscript{60}", 
        "Pr\\textsubscript{70}", 
        "Pr\\textsubscript{80}", 
        "Pr\\textsubscript{90}", 
        "IoU" 
    ]
    
):
    template = get_label_tab_template(
        model_name=model_name, dataset_name=dataset_name
    )
    print(template.split("#")[0])
    print_str = ""
    for col_idx, col_name in enumerate(col_names):
        if col_idx == len(col_names) - 1:
            print_str = f"{print_str} {col_name} \\\\"
        else:
            print_str = f"{print_str} {col_name} &"
    print(f"\t\t{print_str}")
    print("\t\t\midrule")
    
    for class_name, class_data in label_results_dict.items():
        class_name_ = class_name.replace('_', '\_')
        print_str = f"\\texttt{{{class_name_}}} \t&"
        for metric_idx, (metric_name, metric_val) in enumerate(class_data.items()):
            if metric_idx == len(class_data.items()) - 1:
                print_str = f"{print_str} {metric_val} \t \\\\"
            else:
                print_str = f"{print_str} {metric_val} \t &"
        print(f"\t\t{print_str}")
    print(template.split("#")[-1])   


def get_per_label_tables_all_models(data_dict, dataset_name):
    for model_name, label_results_dict in data_dict.items():
        per_label_table(
            label_results_dict=label_results_dict,
            model_name=model_name,
            dataset_name=dataset_name
        )
        print('\n' * 2)

def save_dict_as_json(data_dict: dict, dump_file_path: str):
    print(f"saving to : {dump_file_path}")
    with open(dump_file_path, 'w') as dump_file:
        json.dump(data_dict, dump_file, indent=4)
        
def load_json_file(json_file_path: str) -> dict:
    with open(json_file_path, 'r') as file:
        return json.load(file)

def group_data_by_class_label(label_results_dict):
    grouped_dict = {}
    for model_name, label_data in label_results_dict.items():
        for label_name, label_metrics in label_data.items():
            if label_name not in grouped_dict.keys():
                grouped_dict[label_name] = {}
            grouped_dict[label_name][model_name] = label_metrics["IoU"]
    return grouped_dict

def per_label_per_model_std(
    label_results_dict, 
    dataset_name,
    template = LABEL_MODEL_TABLE_TEMPLATE
):
    
    model_names = list(label_results_dict.keys())
    template = get_label_model_template(
        dataset_name=dataset_name,
        model_names=model_names
    )
    print(template.split("#")[0])
    label_results_dict = group_data_by_class_label(
        label_results_dict=label_results_dict
    )
    for class_name, class_data in label_results_dict.items():
        class_name_ = class_name.replace('_', '\_')
        print_str = f"\\texttt{{{class_name_}}} \t&"
        for model_idx, (model_name, iou_val) in enumerate(class_data.items()):
            if model_idx == len(class_data.items()) - 1:
                print_str = f"{print_str} {iou_val} \t \\\\"
            else:
                print_str = f"{print_str} {iou_val} \t &"
        print(f"\t\t{print_str}")
    print(template.split("#")[-1])   

def per_label_per_model(
    label_results_dict, 
    dataset_name,
    template = LABEL_MODEL_TABLE_TEMPLATE,
    file_path = None
):
    if file_path is None:
        return per_label_per_model_std(
            label_results_dict=label_results_dict,
            dataset_name=dataset_name,
            template=template,
        )
    with open(file_path, 'w') as table_file:
        model_names = list(label_results_dict.keys())
        template = get_label_model_template(
            dataset_name=dataset_name,
            model_names=model_names
        )
        table_file.write(f"{template.split('#')[0]}\n")
        label_results_dict = group_data_by_class_label(
            label_results_dict=label_results_dict
        )
        for class_name, class_data in label_results_dict.items():
            class_name_ = class_name.replace('_', '\_')
            print_str = f"\\texttt{{{class_name_}}} \t&"
            for model_idx, (model_name, iou_val) in enumerate(class_data.items()):
                if model_idx == len(class_data.items()) - 1:
                    print_str = f"{print_str} {iou_val} \t \\\\"
                else:
                    print_str = f"{print_str} {iou_val} \t &"
            table_file.write(f"\t\t{print_str}\n")
        table_file.write(f"{template.split('#')[-1]}\n")  
    
def main():
    file_path = "my_projects/test_results/irl_vision_cat/data/irl_vision_cat_per_label_results.json"
    data_dict = load_json_file(json_file_path=file_path)
    per_label_per_model(
        label_results_dict=data_dict,
        dataset_name="IRL Vision Cat"
    )


if __name__ == '__main__':
    main()
