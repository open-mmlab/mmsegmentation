import os
import subprocess
from my_projects.scripts.parse_model_data import get_all_model_dicts
from copy import deepcopy
import shutil 
project_list = [
    "bisenetv1",
    "mask2former",
    "maskformer",
    "segformer",
    "segnext"
]

model_dicts = get_all_model_dicts(project_list=project_list)
for model_dict in model_dicts:
    print("-" * 80)
    for key, val in model_dict.items():
        print(f"{key} : {val}")
    print()
print(len(model_dicts))
# just the ones tested (drop other backbones)

relevant = []
for method in os.listdir("my_projects/best_models/selection"):
    # only once
    if "hots-v1" in method:
        continue
    [name, backbone] = method.split("_")[:2]
    relevant.append(f"{name}_{backbone}")

model_dicts_, model_dicts = model_dicts, []
for model_dict in model_dicts_:
    if model_dict['Backbone'] == "R-18-D32 (4x8)":
        continue
    for method in relevant:
        if method in model_dict['config']:
            model_dicts.append(model_dict)

for model_dict in model_dicts:
    print("-" * 80)
    for key, val in model_dict.items():
        print(f"{key} : {val}")
    print()
print(len(model_dicts))

for model_dict in model_dicts:
    cfg_src_path = model_dict['config']
    _, cfg_name = os.path.split(cfg_src_path)
    method_name = cfg_name.replace(".py", "")
    method_path = os.path.join(
        "my_projects/best_models/selection_trained",
        method_name
    )
    if not os.path.exists(method_path):
        os.mkdir(method_path)
    cfg_target_path = os.path.join(
        method_path,
        cfg_name
    )
    shutil.copyfile(cfg_src_path, cfg_target_path)
    mim_cmd = f"mim download mmseg --config {method_name} --dest {method_path}"
    # # call_list = ["conda", "activate", "openmmlab", "&&"]
    call_list = []
    for cmd in mim_cmd.split():
        call_list.append(cmd)
    subprocess.call(call_list)
    weights = [
        file_name for file_name in os.listdir(method_path) 
               if ".pth" in file_name
    ][0]
    weights = os.path.join(method_path, weights)
    os.rename(
        weights, 
        os.path.join(
            method_path,
            "weights.pth"
        )
    )
    