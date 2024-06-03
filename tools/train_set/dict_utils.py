
import yaml
import os
from pathlib import Path
from typing import Union
from mmengine import ConfigDict, Config
import queue as queue_

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
            "num_classes"   :   46
        }
}

alias_dict = {
    
}


class TrimData:
    excluded_model_names = [
        "beit-base_upernet_1xb2-5epochs_hots-v1-480x480",
        "beit-base_upernet_1xb2-5epochs_hots-v1-480x480",
        "apcnet_r101-d8_1xb2-pre-cityscapes-5epochs_hots-v1-480x480"
    ]

    accepted_dataset_list = [
        "Cityscapes",
        "ADE20K",
        "Pascal VOC 2012 + Aug",
        "Pascal Context",
        "Pascal Context 59",
        "COCO-Stuff 10k",
        "COCO-Stuff 164k"
    ]
    

    excluded_method_names = [
        "ann_r101-d8",
        "apcnet_r101-d8",
        "beit-base_upernet",
        "beit-large_upernet",
        "bisenetv1_r18-d32",
        "bisenetv1_r50-d32",
        "ccnet_r101-d8",
        "convnext-large_upernet",
        "convnext-xlarge_upernet",
        "danet_r101-d8",
        "ddrnet_23-slim_in1k-pre", # TODO temp
        "ddrnet_23_in1k-pre", # TODO temp
        "deeplabv3_r101b-d8",
        "deeplabv3_r101-d8",
        "deeplabv3plus_r50-d8",
        "deeplabv3plus_r50b-d8",
        "deeplabv3plus_r101-d8",
        "deeplabv3plus_r101b-d8",
        "ddeeplabv3plus_r101-d16-mg124",
        "dmnet_r101-d8",
        "dnl_r50-d8",
        "dnl_r101-d8"
        
        
    ]


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

         

# config_bases =  {
#                     "convnext"  :
#                         {
#                             "algorithm_names"   : 
#                                 [
#                                     "convnext-tiny"
#                                 ],
#                             "backbones"         :
#                                 [
#                                     "upernet"
#                                 ]
#                         }
#                         ,
#                     "ddrnet"    :
#                         {
#                             "algorithm_names"   : 
#                                 [
#                                     "ddrnet"
#                                 ],
#                             "backbones"         :
#                                 [
#                                     "23-slim"
#                                 ]
#                         }
#                         ,
#                     "deeplabv3" :
#                         {
#                             "algorithm_names"   : 
#                                 [
#                                     "deeplabv3"
#                                 ],
#                             "backbones"         :
#                                 [
#                                     "r18-d8",
#                                     "r18b-d8"
#                                 ]  
#                         }
#                         ,
#                     "deeplabv3plus" :
#                         {
#                             "algorithm_names"   : 
#                                 [
#                                     "deeplabv3plus"
#                                 ],
#                             "backbones"         :
#                                 [
#                                     "r18-d8",
#                                     "r18b-d8"
#                                 ]     
#                         }
#                         ,
#                     "fastscnn"  : 
#                         {
#                             "algorithm_names"   : 
#                                 [
#                                     "fcn"
#                                 ],
#                             "backbones"         :
#                                 [
#                                     "fastscnn"
#                                 ]
#                         }
#                         ,
#                     "fcn"  : 
#                         {
#                             "algorithm_names"   : 
#                                 [
#                                     "fcn",
#                                     "fcn-d6"
#                                 ],
#                             "backbones"         :
#                                 [
#                                     "r18-d8",
#                                     "r18b-d8",
#                                     "r50-d16",
#                                     "r50b-d16",
#                                     "r101-d16",
#                                     "r101b-d16"     
#                                 ]
#                         }
#                         ,
#                     "hrnet"  : 
#                         {
#                             "algorithm_names"   : 
#                                 [
#                                     "fcn"
#                                 ],
#                             "backbones"         :
#                                 [
#                                     "hr18",
#                                     "hr18s"
                                    
#                                 ]                                    
#                         }
#                         ,
#                     "icnet"  : 
#                         {
#                             "algorithm_names"   : 
#                                 [
#                                     "icnet"
#                                 ],
#                             "backbones"         :
#                                 [
#                                     "r18-d8",
#                                     "r18-d8-in1k-pre",
#                                     "r50-d8",
#                                     "r50-d8-in1k-pre",
#                                     "r101-d8",
#                                     "r101-d8-in1k-pre"
                                    
#                                 ]       
#                         }
#                         ,
#                     "mask2former"  : 
#                         {
#                             "algorithm_names"   : 
#                                 [
#                                     "mask2former"
#                                 ],
#                             "backbones"         :
#                                 [
#                                     "r50",
#                                     "r101",
#                                     "swin-t",
#                                     "swin-s"
                                    
#                                 ]
#                         }
#                         ,
#                     "maskformer"  : 
#                         {
#                             "algorithm_names"   : 
#                                 [
#                                     "maskformer"
#                                 ],
#                             "backbones"         :
#                                 [
#                                     "r50-d32",
#                                     "r101-d32",
#                                     "swin-t",
#                                     "swin-s"
                                    
#                                 ]
#                         },
#                     "mobilenet_v2"  :
#                         {
#                             "algorithm_names"   : 
#                                 [
#                                     "fcn",
#                                     "pspnet",
#                                     "deeplabv3",
#                                     "deeplabv3plus"
#                                 ],
#                             "backbones"         :
#                                 [
#                                     "mobilenet-v2-d8"
                                    
#                                 ]
#                         },
#                     "pspnet"        :
#                         {
#                              "algorithm_names"   : 
#                                 [
#                                     "pspnet"
#                                 ],
#                             "backbones"         :
#                                 [
#                                     "r18-d8",
#                                     "r18b-d8",
#                                     "r50-d32",
#                                     "r50b-d32",
#                                     "r50-d32-rsb",
#                                 ]
#                         },
#                     "segformer"  : 
#                         {
#                             "algorithm_names"   : 
#                                 [
#                                     "segformer"
#                                 ],
#                             "backbones"         :
#                                 [
#                                     "mit-b0",
#                                     "mit-b1",
#                                     "mit-b2",
#                                     "mit-b3"
                                    
#                                 ]
#                         }
#                         ,
#                     "segmenter"  : 
#                         {
#                             "algorithm_names"   : 
#                                 [
#                                     "segmenter-mask",
#                                     "segmenter-fcn"
#                                 ],
#                             "backbones"         :
#                                 [
#                                     "vit-t",
#                                     "vit-s",
#                                     "vit-b",
                                    
#                                 ]  
#                         }
                        
# }

# # add bigger models with a exception field: check for key in processing
# method_files = {
#         "convnext-tiny_upernet"         
#             :
#                 {
#                     "base_file_path"        :       "configs/convnext/convnext-tiny_upernet_8xb2-amp-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/upernet_convnext_tiny_fp16_512x512_160k_ade20k_20220227_124553-cad485de.pth"  
#                             }
#                         ]
#                 }
#             ,
#         "ddrnet_23-slim"                
#             :
#                 {
#                     "base_file_path"        :       "configs/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024_20230426_145312-6a5e5174.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "deeplabv3_mobilenet-v2-d8"
#             :
#                 {
#                     "base_file_path"        :       "configs/mobilenet_v2/mobilenet-v2-d8_deeplabv3_4xb4-160k_ade20k-512x512.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/deeplabv3_m-v2-d8_512x512_160k_ade20k_20200825_223255-63986343.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "deeplabv3_r18-d8"     
#             :
#                 {
#                     "base_file_path"        :       "configs/deeplabv3/deeplabv3_r18-d8_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/deeplabv3_r18-d8_512x1024_80k_cityscapes_20201225_021506-23dffbe2.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "deeplabv3_r18b-d8"     
#             :
#                 {
#                     "base_file_path"        :       "configs/deeplabv3/deeplabv3_r18b-d8_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/deeplabv3_r18b-d8_512x1024_80k_cityscapes_20201225_094144-46040cef.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "deeplabv3plus_mobilenet-v2-d8"
#             :
#                 {
#                     "base_file_path"        :       "configs/mobilenet_v2/mobilenet-v2-d8_deeplabv3plus_4xb4-160k_ade20k-512x512.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/deeplabv3plus_m-v2-d8_512x512_160k_ade20k_20200825_223255-465a01d4.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "deeplabv3plus_r18-d8" 
#             :
#                 {
#                     "base_file_path"        :       "configs/deeplabv3plus/deeplabv3plus_r18-d8_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "deeplabv3plus_r18b-d8" 
#             :
#                 {
#                     "base_file_path"        :       "configs/deeplabv3plus/deeplabv3plus_r18b-d8_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/deeplabv3plus_r18b-d8_512x1024_80k_cityscapes_20201226_090828-e451abd9.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "fcn_fastscnn" 
#             :
#                 {
#                     "base_file_path"        :       "configs/fastscnn/fast_scnn_8xb4-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853-0cec9937.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "fcn_mobilenet-v2-d8"
#         :
#             {
#                "base_file_path"        :       "configs/mobilenet_v2/mobilenet-v2-d8_fcn_4xb4-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/fcn_m-v2-d8_512x512_160k_ade20k_20200825_214953-c40e1095.pth"  
#                             }
#                         ] 
#             }
#             ,
#         "fcn_r18-d8" 
#             :
#                 {
#                     "base_file_path"        :       "configs/fcn/fcn_r18-d8_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/fcn_r18-d8_512x1024_80k_cityscapes_20201225_021327-6c50f8b4.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "fcn_r18b-d8" 
#             :
#                 {
#                     "base_file_path"        :       "configs/fcn/fcn_r18b-d8_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/fcn_r18b-d8_512x1024_80k_cityscapes_20201225_230143-92c0f445.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "fcn-d6_r50-d16" 
#             :
#                 {
#                     "base_file_path"        :       "configs/fcn/fcn-d6_r50-d16_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/fcn_d6_r50-d16_512x1024_40k_cityscapes_20210305_130133-98d5d1bc.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "fcn-d6_r50b-d16" 
#             :
#                 {
#                     "base_file_path"        :       "configs/fcn/fcn-d6_r50b-d16_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/fcn_d6_r50b-d16_512x1024_80k_cityscapes_20210311_125550-6a0b62e9.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "fcn-d6_r101-d16" 
#             :
#                 {
#                     "base_file_path"        :       "configs/fcn/fcn-d6_r101-d16_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/fcn_d6_r101-d16_512x1024_40k_cityscapes_20210305_130337-9cf2b450.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "fcn-d6_r101b-d16" 
#             :
#                 {
#                     "base_file_path"        :       "configs/fcn/fcn-d6_r101b-d16_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/fcn_d6_r101b-d16_512x1024_80k_cityscapes_20210311_144305-3f2eb5b4.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "fcn_hr18" 
#             :
#                 {
#                     "base_file_path"        :       "configs/hrnet/fcn_hr18_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/fcn_hr18_512x1024_80k_cityscapes_20200601_223255-4e7b345e.pth"  
#                             }
#                             ,
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/fcn_hr18_512x512_80k_ade20k_20210827_114910-6c9382c0.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "fcn_hr18s" 
#             :
#                 {
#                     "base_file_path"        :       "configs/hrnet/fcn_hr18s_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/fcn_hr18s_512x1024_80k_cityscapes_20200601_202700-1462b75d.pth"  
#                             }
#                             ,
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/fcn_hr18s_512x512_80k_ade20k_20200614_144345-77fc814a.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "icnet_r50-d8"         
#             : 
#                 {
#                     "base_file_path"        :       "configs/icnet/icnet_r50-d8_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/icnet_r50-d8_832x832_160k_cityscapes_20210925_232612-a95f0d4e.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "icnet_r50-d8-in1k-pre"         
#             : 
#                 {
#                     "base_file_path"        :       "configs/icnet/icnet_r50-d8-in1k-pre_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/icnet_r50-d8_in1k-pre_832x832_160k_cityscapes_20210926_042715-ce310aea.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "icnet_r18-d8"         
#             : 
#                 {
#                     "base_file_path"        :       "configs/icnet/icnet_r18-d8_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/icnet_r18-d8_832x832_160k_cityscapes_20210925_230153-2c6eb6e0.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "icnet_r18-d8-in1k-pre"         
#             : 
#                 {
#                     "base_file_path"        :       "configs/icnet/icnet_r18-d8-in1k-pre_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/icnet_r18-d8_in1k-pre_832x832_160k_cityscapes_20210926_052702-619c8ae1.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "icnet_r101-d8"         
#             : 
#                 {
#                     "base_file_path"        :       "configs/icnet/icnet_r101-d8_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/icnet_r101-d8_832x832_160k_cityscapes_20210926_092350-3a1ebf1a.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "icnet_r101-d8-in1k-pre"         
#             : 
#                 {
#                     "base_file_path"        :       "configs/icnet/icnet_r18-d8-in1k-pre_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/icnet_r101-d8_in1k-pre_832x832_160k_cityscapes_20210925_232612-9484ae8a.pth"
#                             }
#                         ]
#                 }
#                 ,
#         "mask2former_r50"   
#             : 
#                 {
#                     "base_file_path"        :       "configs/mask2former/mask2former_r50_8xb2-20k_HOTS_v1-512x512.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/mask2former_r50_8xb2-160k_ade20k-512x512_20221204_000055-2d1f55f1.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "mask2former_r101"   
#             : 
#                 {
#                     "base_file_path"        :       "configs/mask2former/mask2former_r101_8xb2-20k_HOTS_v1-512x512.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/mask2former_r101_8xb2-160k_ade20k-512x512_20221203_233905-b7135890.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "mask2former_swin-t"   
#             : 
#                 {
#                     "base_file_path"        :       "configs/mask2former/mask2former_swin-t_8xb2-20k_HOTS_v1-512x512.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/mask2former_swin-t_8xb2-160k_ade20k-512x512_20221203_234230-7d64e5dd.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "mask2former_swin-s"   
#             : 
#                 {
#                     "base_file_path"        :       "configs/mask2former/mask2former_swin-s_8xb2-20k_HOTS_v1-512x512.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/mask2former_swin-s_8xb2-160k_ade20k-512x512_20221204_143905-e715144e.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "maskformer_r50-d32"   
#             : 
#                 {
#                     "base_file_path"        :       "configs/maskformer/maskformer_r50-d32_8xb2-20k_HOTS_v1-512x512.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/maskformer_r50-d32_8xb2-160k_ade20k-512x512_20221030_182724-3a9cfe45.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "maskformer_r101-d32"   
#             : 
#                 {
#                     "base_file_path"        :       "configs/maskformer/maskformer_r101-d32_8xb2-20k_HOTS_v1-512x512.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/maskformer_r101-d32_8xb2-160k_ade20k-512x512_20221031_223053-84adbfcb.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "maskformer_swin-t"   
#             : 
#                 {
#                     "base_file_path"        :       "configs/maskformer/maskformer_swin-t_upernet_8xb2-20k_HOTS_v1-512x512.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/maskformer_swin-t_upernet_8xb2-160k_ade20k-512x512_20221114_232813-f14e7ce0.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "maskformer_swin-s"   
#             : 
#                 {
#                     "base_file_path"        :       "configs/maskformer/maskformer_swin-s_upernet_8xb2-20k_HOTS_v1-512x512.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/maskformer_swin-s_upernet_8xb2-160k_ade20k-512x512_20221115_114710-723512c7.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "pspnet_mobilenet-v2-d8"
#             :
#                 {
#                     "base_file_path"        :       "configs/mobilenet_v2/mobilenet-v2-d8_pspnet_4xb4-160k_ade20k-512x512.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/pspnet_m-v2-d8_512x512_160k_ade20k_20200825_214953-f5942f7a.pth"  
#                             }
#                         ]
#                 },
#         "pspnet_r18-d8"
#             :
#                 {
#                     "base_file_path"        :       "configs/pspnet/pspnet_r18-d8_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/pspnet_r18-d8_512x1024_80k_cityscapes_20201225_021458-09ffa746.pth"  
#                             }
#                         ]
#                 },
#         "pspnet_r18b-d8"
#             :
#                 {
#                     "base_file_path"        :       "configs/pspnet/pspnet_r18b-d8_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/pspnet_r18b-d8_512x1024_80k_cityscapes_20201226_063116-26928a60.pth"  
#                             }
#                         ]
#                 },
#         "pspnet_r50-d32"
#             :
#                 {
#                     "base_file_path"        :       "configs/pspnet/pspnet_r50-d32_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/pspnet_r50-d32_512x1024_80k_cityscapes_20220316_224840-9092b254.pth"  
#                             }
#                         ]
#                 },
#         "pspnet_r50b-d32"
#             :
#                 {
#                     "base_file_path"        :       "configs/pspnet/pspnet_r50b-d32_4xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/pspnet_r50b-d32_512x1024_80k_cityscapes_20220311_152152-23bcaf8c.pth"  
#                             }
#                         ]
#                 },
#                 "pspnet_r50-d32-rsb"
#             :
#                 {
#                     "base_file_path"        :       "configs/pspnet/pspnet_r50-d32_rsb_4xb2-adamw-20k_HOTS_v1-640x480.py ",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "cityscapes",
#                                 "path"              :       "checkpoints/pspnet_r50-d32_rsb-pretrain_512x1024_adamw_80k_cityscapes_20220316_141229-dd9c9610.pth"  
#                             }
#                         ]
#                 },
#         "segformer_mit-b0"     
#             : 
#                 {
#                     "base_file_path"        :       "configs/segformer/segformer_mit-b0_8xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "segformer_mit-b1"     
#             : 
#                 {
#                     "base_file_path"        :       "configs/segformer/segformer_mit-b1_8xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "segformer_mit-b2"     
#             : 
#                 {
#                     "base_file_path"        :       "configs/segformer/segformer_mit-b2_8xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/segformer_mit-b2_512x512_160k_ade20k_20210726_112103-cbd414ac.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "segformer_mit-b3"     
#             : 
#                 {
#                     "base_file_path"        :       "configs/segformer/segformer_mit-b3_8xb2-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/segformer_mit-b3_512x512_160k_ade20k_20210726_081410-962b98d2.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "segmenter-mask_vit-t"     
#             : 
#                 {
#                     "base_file_path"        :       "configs/segmenter/segmenter_vit-t_mask_8xb1-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/segmenter_vit-t_mask_8x1_512x512_160k_ade20k_20220105_151706-ffcf7509.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "segmenter-mask_vit-s"     
#             : 
#                 {
#                     "base_file_path"        :       "configs/segmenter/segmenter_vit-s_mask_8xb1-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/segmenter_vit-s_mask_8x1_512x512_160k_ade20k_20220105_151706-511bb103.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "segmenter-mask_vit-b"     
#             : 
#                 {
#                     "base_file_path"        :       "configs/segmenter/segmenter_vit-b_mask_8xb1-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/segmenter_vit-b_mask_8x1_512x512_160k_ade20k_20220105_151706-bc533b08.pth"  
#                             }
#                         ]
#                 }
#                 ,
#         "segmenter-fcn_vit-s"     
#             : 
#                 {
#                     "base_file_path"        :       "configs/segmenter/segmenter_vit-s_fcn_8xb1-20k_HOTS_v1-640x480.py",
#                     "checkpoints"           :       
#                         [
#                             {
#                                 "dataset_name"      :       "ade20k",
#                                 "path"              :       "checkpoints/segmenter_vit-s_linear_8x1_512x512_160k_ade20k_20220105_151713-39658c46.pth"  
#                             }
#                         ]
#                 }
# }
