import os

def gen_iterations(cfg_dir_path, cfg_name, iters = ["20k", "40k", "80k", "160k"]):
    existing_iter_file = "20k"
    for iter in iters:
        if iter in cfg_name:
            existing_iter_file = iter
    for iter in iters:
        if iter == existing_iter_file:
            continue
        new_cfg = cfg_name.replace(existing_iter_file, iter)
        source_path = os.path.join(cfg_dir_path, cfg_name)
        target_path = os.path.join(cfg_dir_path, new_cfg)
        os.system(f"cp {source_path} {target_path}")
        



def get_all_iters_of_cfg(cfg_name, iters = ["20k", "40k", "80k", "160k"]):
    names = [cfg_name]
    for iter in iters:
        if iter in cfg_name:
            existing_iter_file = iter
    for iter in iters:
        if iter != existing_iter_file:
            names.append(cfg_name.replace(existing_iter_file, iter))
    return names
        
        
def gen_pretrained(cfg_dir_path, cfg_name, pretrained_appendix):
    cfg_names = get_all_iters_of_cfg(cfg_name=cfg_name)
    for cfg_name in cfg_names:
        source_path = os.path.join(cfg_dir_path, cfg_name)
        target_path = os.path.join(cfg_dir_path, 
                                   cfg_name.replace(".py", f"{pretrained_appendix}.py"))
        os.system(f"cp {source_path} {target_path}") 

cfg_dir_path = "configs/fastscnn/"
cfg_name = "pspnet_r18-d8_4xb2-10k_HOTS_v1-640x480.py"

gen_iterations(cfg_dir_path=cfg_dir_path, cfg_name=cfg_name, iters=["20k"])

# gen_pretrained(
#                     cfg_dir_path = cfg_dir_path, 
#                     cfg_name = cfg_name, 
#                     pretrained_appendix="_pretrained_ade20k"
#                )