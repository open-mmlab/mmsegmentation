
import yaml
import os
from pathlib import Path
from typing import Union
from mmengine import ConfigDict, Config
import queue as queue_
import copy

DEFAULT_CONFIG_ROOT_PATH = "/media/ids/Ubuntu files/mmsegmentation/configs"
BATCH_SIZE_DEFAULT = 2
N_GPU_DEFAULT = 1 
VAL_INTERVAL_EPOCH_DEFAULT = 1
VAL_INTERVAL_ITERATIONS_DEFAULT = 1000   
N_ITERATIONS_DEFAULT = 20000 

dataset_info = {
    "hots-v1"   :     
        {
            "cfg_path"      :   "configs/_base_/datasets/hots_v1_640x480.py",
            "num_classes"   :   47,
            "class_weight"  :   None 
        },
    "irl_vision_sim"    :
        {
            "cfg_path"      :   "configs/_base_/datasets/irl_vision_sim_640x480.py",
            "num_classes"   :   72,
            "class_weight"  :   None 
        }
}

alias_dict = {
    
}


class TrimData:
    

    accepted_dataset_list = [
        "COCO-Stuff 164k",
        "COCO-Stuff 10k",
        "ADE20K",
        "Pascal VOC 2012 + Aug",
        "Pascal Context",
        "Pascal Context 59",
        "Cityscapes" 
    ]
    
    excluded_method_names = [
        "ann_r101-d8",
        "apcnet_r101-d8",
        "beit-base_upernet",
        "beit-large_upernet",
        "bisenetv1_r18-d32",  # only take pretrained vars (see repo)
        "bisenetv1_r50-d32",
        "bisenetv1_r101-d32",
        "ccnet_r101-d8",
        "convnext-large_upernet",
        "convnext-xlarge_upernet",
        "danet_r101-d8",
        "deeplabv3_r18b-d8", # take out b versions due to redundancy
        "deeplabv3_r50b-d8",
        "deeplabv3_r101b-d8",
        "deeplabv3_r101-d8",
        # "deeplabv3plus_r50-d8",
        "deeplabv3plus_r50b-d8",
        "deeplabv3plus_r101-d8",
        "deeplabv3plus_r101b-d8",
        "deeplabv3plus_r101-d16-mg124",
        "ddeeplabv3plus_r101-d16-mg124",
        "deeplabv3plus_r18b-d8",
        "dmnet_r50-d8",
        "dmnet_r101-d8",
        "dnl_r50-d8",
        "dnl_r101-d8",
        "dpt_vit-b16",
        "emanet_r50-d8",
        "eemanet_r50-d8",
        "emanet_r101-d8",
        "encnet_r50-d8",
        "encnet_r101-d8",
        "fastfcn_r50-d32_jpu_enc",
        "fcn_r18b-d8", # for these 3 I ignore the worst of the options (b/no b)
        "fcn_r50-d8",
        "fcn_r101-d8",
        "fcn_r101b-d8",
        "fcn-d6_r50b", # best of b or !b
        "fcn-d6_r101b",
        "gcnet_r101-d8",
        "icnet_r18-d8", # better options are pretrained
        "icnet_r50-d8",
        "icnet_r101-d8",
        "isanet_r50-d8",
        "isanet_r101-d8",
        "knet-s3_swin-l_upernet",
        "mae-base_upernet",
        "mask2former_swin-s",
        "mask2former_swin-b-in22k-384x384-pre",
        "mask2former_swin-l-in22k-384x384-pre",
        "mask2former_swin-b-in1k-384x384-pre",
        "maskformer_swin-s_upernet",
        "mobilenet-v3-d8-scratch_lraspp",
        "mobilenet-v3-d8-scratch-s_lraspp",
        "nonlocal_r101-d8",
        "ocrnet_hr48",
        "pidnet-s",
        "pidnet-m",
        "pidnet-l",
        "psanet_r101-d8",
        "pspnet_r18b-d8",
        "pspnet_r50b-d8",
        "pspnet_r101-d8",
        "pspnet_r101b-d8",
        "resnest_s101-d8_fcn",
        "resnest_s101-d8_pspnet",
        "resnest_s101-d8_deeplabv3",
        "resnest_s101-d8_deeplabv3plus",
        "san-vit-b16_coco-stuff164k-640x",
        "san-vit-l14_coco-stuff164k-640x",
        "segformer_mit-b4",
        "segformer_mit-b5",
        "segmenter_vit-l_mask",
        "segnext_mscan-l",
        "setr_vit-l_naive",
        "setr_vit-l_pup",
        "setr_vit-l_mla",
        "setr_vit-l-mla",
        "swin-base-patch4-window7-in1k-pre_upernet",
        "swin-base-patch4-window12-in1k-384x384-pre_upernet",
        "stdc1",
        "stdc2",
        "swin-base-patch4-window7-in22k-pre_upernet",
        "swin-base-patch4-window12-in22k-384x384-pre_upernet",
        "twins_pcpvt-b_uperhead",
        "twins_pcpvt-l_uperhead",
        "twins_svt-b_uperhead",
        "twins_svt-l_fpn_fpnhead",
        "unet-s5-d16_fcn",
        "vit_vit-b16_mln_upernet",
        "vit_vit-b16-ln_mln_upernet",
        "vit_deit-s16_mln_upernet",
        "vit_deit-b16_upernet",
        "vit_deit-b16_mln_upernet",
        "vit_deit-b16-ln_mln_upernet"
        
        
    ]

    @staticmethod
    def _get_exclusion_lists_from_args(args) -> tuple:
        
        accepted_dataset_list_ = copy.deepcopy(TrimData.accepted_dataset_list)
        excluded_method_names_ = copy.deepcopy(TrimData.excluded_method_names) 
        
        if args.crop_size == (512, 512):
            excluded_method_names_.append("ddrnet_23-slim_in1k-pre")
            excluded_method_names_.append("ddrnet_23_in1k-pre")
        
        return (accepted_dataset_list_, excluded_method_names_)
        
        
class Vertex:
    def __init__(self, item, parent, key = None):
        self.item = item
        self.parent = parent
        self.key = key
    
    def __eq__(self, value: object) -> bool:
        if isinstance(value, self.__class__):
            return value.item == self.item and value.parent == self.parent
        return False

# BFS

def BFS_change_key(cfg, target_key, new_value):
    queue = list()
    vertex = Vertex(cfg, None)
    queue.append(vertex)
    while queue:
        vertex = queue.pop(0)
        children = expand_vertex(vertex)
        if vertex.key == target_key:
            vertex.parent.item[vertex.key] = new_value
        for child in children:
            queue.append(child)
            
        
def expand_vertex(vertex: Vertex) -> list:
    children = []
    if type(vertex.item) is Config or type(vertex.item) is ConfigDict or type(vertex.item) is dict:
        for key, val in vertex.item.items():
            item_ = vertex.item[key]
            children.append(
                Vertex(
                    item=item_,
                    parent=vertex,
                    key=key
                )
            )
    elif type(vertex.item) is list:
        for entry in vertex.item:
            children.append(
                Vertex(
                    item=entry,
                    parent=vertex,
                    key=None
                )
            )
    else:
        return []
    
    return children

def is_leaf(vertex: Vertex) -> bool:
    return bool(expand_vertex(vertex=vertex))

   