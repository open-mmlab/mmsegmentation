import os

dict_list = []
for project_name in os.listdir("."):
    if not os.path.isdir(project_name):
        continue
    if "iter_500.pth" not in os.listdir(project_name):
        continue
    proj_iter_pth = os.path.join(project_name, "iter_500.pth")
    # in iter 500
    for exp_dir in os.listdir(proj_iter_pth):
        exp_dir_path = os.path.join(proj_iter_pth, exp_dir)
        if not os.path.isdir(exp_dir_path):
            continue
        for file_name in os.listdir(exp_dir_path):
            if '.log' not in file_name:
                continue
            log_file_path = os.path.join(exp_dir_path, file_name)
            with open(log_file_path, 'r') as log_file:
                data = log_file.readlines()[-1]
                data = data.split("]")[-1].split()
                
                data_dict = {}
                data_dict["project"] = project_name
                data_dict["file_path"] = log_file_path
                for idx in range(0, len(data) - 1, 2):
                    data_dict[data[idx][:-1]] = data[idx + 1]
                    if data[idx + 1].isnumeric():
                        data_dict[data[idx][:-1]] = data[idx + 1]
                # for key, val in data_dict.items():
                #     print(f"{key} : {val}")
                # print('#' * 40)   
                dict_list.append(data_dict)


req_keys = dict_list[-1].keys()
rm_list = []
print(f"required keys: {req_keys}")
for data_dict in dict_list:
    if req_keys != data_dict.keys():
        
        rm_list.append(data_dict)
    # if 'mIoU' not in data_dict.keys():
        
    #     print(data_dict)

print(f"remove_list:\n {rm_list}")
print()
dict_list = [d_dict for d_dict in dict_list if d_dict not in rm_list]

for d_dict in dict_list:
    for key, val in d_dict.items():
        if key not in ["file_path", "project"]:
            d_dict[key] = float(val)
print(f"{'#' * 60}\n{'#' * 60}\n{'#' * 60}")


# ['aAcc:', '89.0800', 'mIoU:', '10.8400', 'mAcc:', '16.3700', 'mDice:', '13.6600',
# 'mFscore:', '45.8600', 'mPrecision:', '44.7000', 'mRecall:', '16.3700', 'map25:', 
# '13.5800', 'map50:', '8.5500', 'map75:', '6.6700', 'data_time:', '0.0058', 
# 'time:', '10.6031']
from operator import itemgetter
from copy import deepcopy

def get_top_n_metric(dict_list, n = 10, metric ='mIoU', reverse=True):
    # sort_list =  sorted(dict_list, key=itemgetter(metric), reverse=True) 
    sort_list =  sorted(dict_list, key=lambda d : d[metric], reverse=reverse) 
    if len(sort_list) >= n:      
        return sort_list[:n]
    return sort_list

def print_dict_list(dict_list):
    for data_dict in dict_list:
        print('#' * 40)
        for key, val in data_dict.items():
            print(f"{key} : {val}")


selection = ['aAcc', 'mIoU', 'mAcc', 'mDice', 'mFscore', 
             'mPrecision', 'mRecall', 'map50', 'map25', 'map75', 'data_time'
             
            ]

proj_hist = {}
for metric in selection:
    print('#'* 60)
    print(f"\nMETRIC: {metric}")
    print('#'* 60)
    reverse = True 
    if metric == 'data_time':
        reverse = False
    top_10 = get_top_n_metric(
        dict_list=deepcopy(dict_list), n=10, metric=metric, reverse=reverse
    )
    for item in top_10:
        proj_name = item["project"]
        if proj_name in proj_hist.keys():
            proj_hist[proj_name] += 1
        else:
            proj_hist[proj_name] = 1
    
    print_dict_list(top_10)

proj_hist = {k: v for k, v in sorted(proj_hist.items(), key=lambda item : item[1])}
for key, val in proj_hist.items():
    print(f"{key} : {val}")
# print(len(dict_list))
# print("type miou")
# print(type(dict_list[0]['mIoU']))
# for metric in selection:
#     high_score = 0
#     best_model = None
#     for mdl_dict in dict_list:
#         score = mdl_dict[metric]
#         if score > high_score:
#             best_model = mdl_dict
#             high_score = score
#     print(f"METRIC: {metric}\nBEST:")
#     for key, val in best_model.items():
#             print(f"{key} : {val}")
#     print('#'* 60)