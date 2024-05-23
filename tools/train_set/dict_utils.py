N_ITER_DEFAULT = 20000
DEFAULT_BATCH_SIZE = 2
config_bases =  {
                    "convnext"  :
                        {
                            "algorithm_names"   : 
                                [
                                    "convnext-tiny"
                                ],
                            "backbones"         :
                                [
                                    "upernet"
                                ],
                            "train_settings"    :
                                [
                                    {
                                        "n_gpus"        :       1,
                                        "batch_size"    :       DEFAULT_BATCH_SIZE,
                                        "iterations"    :       [N_ITER_DEFAULT]
                                    }
                                ],      
                            "datasets"          :
                                [
                                    "HOTS_v1"
                                ],
                            "crops"             :
                                [
                                    (640, 480)
                                ]  
                        }
                        ,
                    "ddrnet"    :
                        {
                            "algorithm_names"   : 
                                [
                                    "ddrnet"
                                ],
                            "backbones"         :
                                [
                                    "23-slim"
                                ],      
                            "train_settings"    :
                                [
                                    {
                                        "n_gpus"        :       1,
                                        "batch_size"    :       DEFAULT_BATCH_SIZE,
                                        "iterations"    :       [N_ITER_DEFAULT]
                                    }
                                ],      
                            "datasets"          :
                                [
                                    "HOTS_v1"
                                ],
                            "crops"             :
                                [
                                    (640, 480)
                                ]
                        }
                        ,
                    "deeplabv3" :
                        {
                            "algorithm_names"   : 
                                [
                                    "deeplabv3"
                                ],
                            "backbones"         :
                                [
                                    "r18-d8",
                                    "r18b-d8"
                                ],      
                            "train_settings"    :
                                [
                                    {
                                        "n_gpus"        :       1,
                                        "batch_size"    :       DEFAULT_BATCH_SIZE,
                                        "iterations"    :       [N_ITER_DEFAULT]
                                    }
                                ],      
                            "datasets"          :
                                [
                                    "HOTS_v1"
                                ],
                            "crops"             :
                                [
                                    (640, 480)
                                ]
                            
                            
                        }
                        ,
                    "deeplabv3plus" :
                        {
                            "algorithm_names"   : 
                                [
                                    "deeplabv3plus"
                                ],
                            "backbones"         :
                                [
                                    "r18-d8",
                                    "r18b-d8"
                                ],      
                            "train_settings"    :
                                [
                                    {
                                        "n_gpus"        :       1,
                                        "batch_size"    :       DEFAULT_BATCH_SIZE,
                                        "iterations"    :       [N_ITER_DEFAULT]
                                    }
                                ],      
                            "datasets"          :
                                [
                                    "HOTS_v1"
                                ],
                            "crops"             :
                                [
                                    (640, 480)
                                ]
                            
                            
                        }
                        ,
                    "fastscnn"  : 
                        {
                            "algorithm_names"   : 
                                [
                                    "fcn"
                                ],
                            "backbones"         :
                                [
                                    "fastscnn"
                                ],      
                            "train_settings"    :
                                [
                                    {
                                        "n_gpus"        :       1,
                                        "batch_size"    :       DEFAULT_BATCH_SIZE,
                                        "iterations"    :       [N_ITER_DEFAULT]
                                    }
                                ],      
                            "datasets"          :
                                [
                                    "HOTS_v1"
                                ],
                            "crops"             :
                                [
                                    (640, 480)
                                ]    
                        }
                        ,
                    "fcn"  : 
                        {
                            "algorithm_names"   : 
                                [
                                    "fcn",
                                    "fcn-d6"
                                ],
                            "backbones"         :
                                [
                                    "r18-d8",
                                    "r18b-d8",
                                    "r50-d16",
                                    "r50b-d16",
                                    "r101-d16",
                                    "r101b-d16"
                                    
                                ],      
                            "train_settings"    :
                                [
                                    {
                                        "n_gpus"        :       1,
                                        "batch_size"    :       DEFAULT_BATCH_SIZE,
                                        "iterations"    :       [N_ITER_DEFAULT]
                                    }
                                ],      
                            "datasets"          :
                                [
                                    "HOTS_v1"
                                ],
                            "crops"             :
                                [
                                    (640, 480)
                                ]  
                        }
                        ,
                    "hrnet"  : 
                        {
                            "algorithm_names"   : 
                                [
                                    "fcn"
                                ],
                            "backbones"         :
                                [
                                    "hr18",
                                    "hr18s"
                                    
                                ],      
                            "train_settings"    :
                                [
                                    {
                                        "n_gpus"        :       1,
                                        "batch_size"    :       DEFAULT_BATCH_SIZE,
                                        "iterations"    :       [N_ITER_DEFAULT]
                                    }
                                ],      
                            "datasets"          :
                                [
                                    "HOTS_v1"
                                ],
                            "crops"             :
                                [
                                    (640, 480)
                                ]                                    
                        }
                        ,
                    "icnet"  : 
                        {
                            "algorithm_names"   : 
                                [
                                    "icnet"
                                ],
                            "backbones"         :
                                [
                                    "r18-d8",
                                    "r18-d8-in1k-pre",
                                    "r50-d8",
                                    "r50-d8-in1k-pre",
                                    "r101-d8",
                                    "r101-d8-in1k-pre"
                                    
                                ],      
                            "train_settings"    :
                                [
                                    {
                                        "n_gpus"        :       1,
                                        "batch_size"    :       DEFAULT_BATCH_SIZE,
                                        "iterations"    :       [N_ITER_DEFAULT]
                                    }
                                ],      
                            "datasets"          :
                                [
                                    "HOTS_v1"
                                ],
                            "crops"             :
                                [
                                    (640, 480)
                                ]        
                        }
                        ,
                    "mask2former"  : 
                        {
                            "algorithm_names"   : 
                                [
                                    "mask2former"
                                ],
                            "backbones"         :
                                [
                                    "r50",
                                    "r101",
                                    "swin-t",
                                    "swin-s"
                                    
                                ],      
                            "train_settings"    :
                                [
                                    {
                                        "n_gpus"        :       1,
                                        "batch_size"    :       DEFAULT_BATCH_SIZE,
                                        "iterations"    :       [N_ITER_DEFAULT]
                                    }
                                ],      
                            "datasets"          :
                                [
                                    "HOTS_v1"
                                ],
                            "crops"             :
                                [
                                    (512, 512)
                                ]
                        }
                        ,
                    "maskformer"  : 
                        {
                            "algorithm_names"   : 
                                [
                                    "maskformer"
                                ],
                            "backbones"         :
                                [
                                    "r50-d32",
                                    "r101-d32",
                                    "swin-t",
                                    "swin-s"
                                    
                                ],      
                            "train_settings"    :
                                [
                                    {
                                        "n_gpus"        :       1,
                                        "batch_size"    :       DEFAULT_BATCH_SIZE,
                                        "iterations"    :       [N_ITER_DEFAULT]
                                    }
                                ],      
                            "datasets"          :
                                [
                                    "HOTS_v1"
                                ],
                            "crops"             :
                                [
                                    (512, 512)
                                ]
                        },
                    "segformer"  : 
                        {
                            "algorithm_names"   : 
                                [
                                    "segformer"
                                ],
                            "backbones"         :
                                [
                                    "mit-b0",
                                    "mit-b1",
                                    "mit-b2",
                                    "mit-b3"
                                    
                                ],      
                            "train_settings"    :
                                [
                                    {
                                        "n_gpus"        :       1,
                                        "batch_size"    :       DEFAULT_BATCH_SIZE,
                                        "iterations"    :       [N_ITER_DEFAULT]
                                    }
                                ],      
                            "datasets"          :
                                [
                                    "HOTS_v1"
                                ],
                            "crops"             :
                                [
                                    (640, 480)
                                ]
                        }
                        ,
                    "segmenter"  : 
                        {
                            "algorithm_names"   : 
                                [
                                    "segmenter-mask",
                                    "segmenter-fcn"
                                ],
                            "backbones"         :
                                [
                                    "vit-t",
                                    "vit-s",
                                    "vit-b",
                                    
                                ],      
                            "train_settings"    :
                                [
                                    {
                                        "n_gpus"        :       1,
                                        "batch_size"    :       DEFAULT_BATCH_SIZE,
                                        "iterations"    :       [N_ITER_DEFAULT]
                                    }
                                ],      
                            "datasets"          :
                                [
                                    "HOTS_v1"
                                ],
                            "crops"             :
                                [
                                    (640, 480)
                                ]   
                        }
                        
}

# add bigger models with a exception field: check for key in processing
method_files = {
        "convnext-tiny_upernet"         
            :
                {
                    "base_file_path"        :       "configs/convnext/convnext-tiny_upernet_8xb2-amp-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "ade20k",
                                "path"              :       "checkpoints/upernet_convnext_tiny_fp16_512x512_160k_ade20k_20220227_124553-cad485de.pth"  
                            }
                        ]
                }
            ,
        "ddrnet_23-slim"                
            :
                {
                    "base_file_path"        :       "configs/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "cityscapes",
                                "path"              :       "checkpoints/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024_20230426_145312-6a5e5174.pth"  
                            }
                        ]
                }
                ,
        "deeplabv3_r18-d8"     
            :
                {
                    "base_file_path"        :       "configs/deeplabv3/deeplabv3_r18-d8_4xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "cityscapes",
                                "path"              :       "checkpoints/deeplabv3_r18-d8_512x1024_80k_cityscapes_20201225_021506-23dffbe2.pth"  
                            }
                        ]
                }
                ,
        "deeplabv3_r18b-d8"     
            :
                {
                    "base_file_path"        :       "configs/deeplabv3/deeplabv3_r18b-d8_4xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "cityscapes",
                                "path"              :       "checkpoints/deeplabv3_r18b-d8_512x1024_80k_cityscapes_20201225_094144-46040cef.pth"  
                            }
                        ]
                }
                ,
        "deeplabv3plus_r18-d8" 
            :
                {
                    "base_file_path"        :       "configs/deeplabv3plus/deeplabv3plus_r18-d8_4xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "cityscapes",
                                "path"              :       "checkpoints/deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth"  
                            }
                        ]
                }
                ,
        "deeplabv3plus_r18b-d8" 
            :
                {
                    "base_file_path"        :       "configs/deeplabv3plus/deeplabv3plus_r18b-d8_4xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "cityscapes",
                                "path"              :       "checkpoints/deeplabv3plus_r18b-d8_512x1024_80k_cityscapes_20201226_090828-e451abd9.pth"  
                            }
                        ]
                }
                ,
        "fcn_fastscnn" 
            :
                {
                    "base_file_path"        :       "configs/fastscnn/fast_scnn_8xb4-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "cityscapes",
                                "path"              :       "checkpoints/fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853-0cec9937.pth"  
                            }
                        ]
                }
                ,
        "fcn_r18-d8" 
            :
                {
                    "base_file_path"        :       "configs/fcn/fcn_r18-d8_4xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "cityscapes",
                                "path"              :       "checkpoints/fcn_r18-d8_512x1024_80k_cityscapes_20201225_021327-6c50f8b4.pth"  
                            }
                        ]
                }
                ,
        "fcn_r18b-d8" 
            :
                {
                    "base_file_path"        :       "configs/fcn/fcn_r18b-d8_4xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "cityscapes",
                                "path"              :       "checkpoints/fcn_r18b-d8_512x1024_80k_cityscapes_20201225_230143-92c0f445.pth"  
                            }
                        ]
                }
                ,
        "fcn-d6_r50-d16" 
            :
                {
                    "base_file_path"        :       "configs/fcn/fcn-d6_r50-d16_4xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "cityscapes",
                                "path"              :       "checkpoints/fcn_d6_r50-d16_512x1024_40k_cityscapes_20210305_130133-98d5d1bc.pth"  
                            }
                        ]
                }
                ,
        "fcn-d6_r50b-d16" 
            :
                {
                    "base_file_path"        :       "configs/fcn/fcn-d6_r50b-d16_4xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "cityscapes",
                                "path"              :       "checkpoints/fcn_d6_r50b-d16_512x1024_80k_cityscapes_20210311_125550-6a0b62e9.pth"  
                            }
                        ]
                }
                ,
        "fcn-d6_r101-d16" 
            :
                {
                    "base_file_path"        :       "configs/fcn/fcn-d6_r101-d16_4xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "cityscapes",
                                "path"              :       "checkpoints/fcn_d6_r101-d16_512x1024_40k_cityscapes_20210305_130337-9cf2b450.pth"  
                            }
                        ]
                }
                ,
        "fcn-d6_r101b-d16" 
            :
                {
                    "base_file_path"        :       "configs/fcn/fcn-d6_r101b-d16_4xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "cityscapes",
                                "path"              :       "checkpoints/fcn_d6_r101b-d16_512x1024_80k_cityscapes_20210311_144305-3f2eb5b4.pth"  
                            }
                        ]
                }
                ,
        "fcn_hr18" 
            :
                {
                    "base_file_path"        :       "configs/hrnet/fcn_hr18_4xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "cityscapes",
                                "path"              :       "checkpoints/fcn_hr18_512x1024_80k_cityscapes_20200601_223255-4e7b345e.pth"  
                            }
                            ,
                            {
                                "dataset_name"      :       "ade20k",
                                "path"              :       "checkpoints/fcn_hr18_512x512_80k_ade20k_20210827_114910-6c9382c0.pth"  
                            }
                        ]
                }
                ,
        "fcn_hr18s" 
            :
                {
                    "base_file_path"        :       "configs/hrnet/fcn_hr18s_4xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "cityscapes",
                                "path"              :       "checkpoints/fcn_hr18s_512x1024_80k_cityscapes_20200601_202700-1462b75d.pth"  
                            }
                            ,
                            {
                                "dataset_name"      :       "ade20k",
                                "path"              :       "checkpoints/fcn_hr18s_512x512_80k_ade20k_20200614_144345-77fc814a.pth"  
                            }
                        ]
                }
                ,
        "icnet_r50-d8"         
            : 
                {
                    "base_file_path"        :       "configs/icnet/icnet_r50-d8_4xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "cityscapes",
                                "path"              :       "checkpoints/icnet_r50-d8_832x832_160k_cityscapes_20210925_232612-a95f0d4e.pth"  
                            }
                        ]
                }
                ,
        "icnet_r50-d8-in1k-pre"         
            : 
                {
                    "base_file_path"        :       "configs/icnet/icnet_r50-d8-in1k-pre_4xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "cityscapes",
                                "path"              :       "checkpoints/icnet_r50-d8_in1k-pre_832x832_160k_cityscapes_20210926_042715-ce310aea.pth"  
                            }
                        ]
                }
                ,
        "icnet_r18-d8"         
            : 
                {
                    "base_file_path"        :       "configs/icnet/icnet_r18-d8_4xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "cityscapes",
                                "path"              :       "checkpoints/icnet_r18-d8_832x832_160k_cityscapes_20210925_230153-2c6eb6e0.pth"  
                            }
                        ]
                }
                ,
        "icnet_r18-d8-in1k-pre"         
            : 
                {
                    "base_file_path"        :       "configs/icnet/icnet_r18-d8-in1k-pre_4xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "cityscapes",
                                "path"              :       "checkpoints/icnet_r18-d8_in1k-pre_832x832_160k_cityscapes_20210926_052702-619c8ae1.pth"  
                            }
                        ]
                }
                ,
        "icnet_r101-d8"         
            : 
                {
                    "base_file_path"        :       "configs/icnet/icnet_r101-d8_4xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "cityscapes",
                                "path"              :       "checkpoints/icnet_r101-d8_832x832_160k_cityscapes_20210926_092350-3a1ebf1a.pth"  
                            }
                        ]
                }
                ,
        "icnet_r101-d8-in1k-pre"         
            : 
                {
                    "base_file_path"        :       "configs/icnet/icnet_r18-d8-in1k-pre_4xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "cityscapes",
                                "path"              :       "checkpoints/icnet_r101-d8_in1k-pre_832x832_160k_cityscapes_20210925_232612-9484ae8a.pth"
                            }
                        ]
                }
                ,
        "mask2former_r50"   
            : 
                {
                    "base_file_path"        :       "configs/mask2former/mask2former_r50_8xb2-20k_HOTS_v1-512x512.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "ade20k",
                                "path"              :       "checkpoints/mask2former_r50_8xb2-160k_ade20k-512x512_20221204_000055-2d1f55f1.pth"  
                            }
                        ]
                }
                ,
        "mask2former_r101"   
            : 
                {
                    "base_file_path"        :       "configs/mask2former/mask2former_r101_8xb2-20k_HOTS_v1-512x512.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "ade20k",
                                "path"              :       "checkpoints/mask2former_r101_8xb2-160k_ade20k-512x512_20221203_233905-b7135890.pth"  
                            }
                        ]
                }
                ,
        "mask2former_swin-t"   
            : 
                {
                    "base_file_path"        :       "configs/mask2former/mask2former_swin-t_8xb2-20k_HOTS_v1-512x512.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "ade20k",
                                "path"              :       "checkpoints/mask2former_swin-t_8xb2-160k_ade20k-512x512_20221203_234230-7d64e5dd.pth"  
                            }
                        ]
                }
                ,
        "mask2former_swin-s"   
            : 
                {
                    "base_file_path"        :       "configs/mask2former/mask2former_swin-s_8xb2-20k_HOTS_v1-512x512.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "ade20k",
                                "path"              :       "checkpoints/mask2former_swin-s_8xb2-160k_ade20k-512x512_20221204_143905-e715144e.pth"  
                            }
                        ]
                }
                ,
        "maskformer_r50-d32"   
            : 
                {
                    "base_file_path"        :       "configs/maskformer/maskformer_r50-d32_8xb2-20k_HOTS_v1-512x512.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "ade20k",
                                "path"              :       "checkpoints/maskformer_r50-d32_8xb2-160k_ade20k-512x512_20221030_182724-3a9cfe45.pth"  
                            }
                        ]
                }
                ,
        "maskformer_r101-d32"   
            : 
                {
                    "base_file_path"        :       "configs/maskformer/maskformer_r101-d32_8xb2-20k_HOTS_v1-512x512.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "ade20k",
                                "path"              :       "checkpoints/maskformer_r101-d32_8xb2-160k_ade20k-512x512_20221031_223053-84adbfcb.pth"  
                            }
                        ]
                }
                ,
        "maskformer_swin-t"   
            : 
                {
                    "base_file_path"        :       "configs/maskformer/maskformer_swin-t_upernet_8xb2-20k_HOTS_v1-512x512.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "ade20k",
                                "path"              :       "checkpoints/maskformer_swin-t_upernet_8xb2-160k_ade20k-512x512_20221114_232813-f14e7ce0.pth"  
                            }
                        ]
                }
                ,
        "maskformer_swin-s"   
            : 
                {
                    "base_file_path"        :       "configs/maskformer/maskformer_swin-s_upernet_8xb2-20k_HOTS_v1-512x512.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "ade20k",
                                "path"              :       "checkpoints/maskformer_swin-s_upernet_8xb2-160k_ade20k-512x512_20221115_114710-723512c7.pth"  
                            }
                        ]
                }
                ,
        
        "segformer_mit-b0"     
            : 
                {
                    "base_file_path"        :       "configs/segformer/segformer_mit-b0_8xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "ade20k",
                                "path"              :       "checkpoints/segformer_mit-b0_512x512_160k_ade20k_20210726_101530-8ffa8fda.pth"  
                            }
                        ]
                }
                ,
        "segformer_mit-b1"     
            : 
                {
                    "base_file_path"        :       "configs/segformer/segformer_mit-b1_8xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "ade20k",
                                "path"              :       "checkpoints/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth"  
                            }
                        ]
                }
                ,
        "segformer_mit-b2"     
            : 
                {
                    "base_file_path"        :       "configs/segformer/segformer_mit-b2_8xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "ade20k",
                                "path"              :       "checkpoints/segformer_mit-b2_512x512_160k_ade20k_20210726_112103-cbd414ac.pth"  
                            }
                        ]
                }
                ,
        "segformer_mit-b3"     
            : 
                {
                    "base_file_path"        :       "configs/segformer/segformer_mit-b3_8xb2-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "ade20k",
                                "path"              :       "checkpoints/segformer_mit-b3_512x512_160k_ade20k_20210726_081410-962b98d2.pth"  
                            }
                        ]
                }
                ,
        "segmenter-mask_vit-t"     
            : 
                {
                    "base_file_path"        :       "configs/segmenter/segmenter_vit-t_mask_8xb1-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "ade20k",
                                "path"              :       "checkpoints/segmenter_vit-t_mask_8x1_512x512_160k_ade20k_20220105_151706-ffcf7509.pth"  
                            }
                        ]
                }
                ,
        "segmenter-mask_vit-s"     
            : 
                {
                    "base_file_path"        :       "configs/segmenter/segmenter_vit-s_mask_8xb1-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "ade20k",
                                "path"              :       "checkpoints/segmenter_vit-s_mask_8x1_512x512_160k_ade20k_20220105_151706-511bb103.pth"  
                            }
                        ]
                }
                ,
        "segmenter-mask_vit-b"     
            : 
                {
                    "base_file_path"        :       "configs/segmenter/segmenter_vit-b_mask_8xb1-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "ade20k",
                                "path"              :       "checkpoints/segmenter_vit-b_mask_8x1_512x512_160k_ade20k_20220105_151706-bc533b08.pth"  
                            }
                        ]
                }
                ,
        "segmenter-fcn_vit-s"     
            : 
                {
                    "base_file_path"        :       "configs/segmenter/segmenter_vit-s_fcn_8xb1-20k_HOTS_v1-640x480.py",
                    "checkpoints"           :       
                        [
                            {
                                "dataset_name"      :       "ade20k",
                                "path"              :       "checkpoints/segmenter_vit-s_linear_8x1_512x512_160k_ade20k_20220105_151713-39658c46.pth"  
                            }
                        ]
                }
}
