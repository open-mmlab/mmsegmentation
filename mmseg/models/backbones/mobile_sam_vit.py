from typing import Tuple, Type, List, Union
from segment_anything.modeling.image_encoder import ImageEncoderViT
from mobile_sam.modeling.tiny_vit_sam import TinyViT

import torch
from torch.nn import functional as F
from functools import partial

from mmseg.registry import MODELS
from mmengine.model.base_module import BaseModule


@MODELS.register_module()
class MobileSAMImageEncoderViT( BaseModule, TinyViT):

    def __init__(self, init_cfg: Union[dict, List[dict], None] = None) -> None:
        BaseModule.__init__(self, init_cfg)
        TinyViT.__init__(
            self,
            img_size=1024,
            in_chans=3,
            num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8

        )

        # freeze all params
        for _, param in self.named_parameters():
            param.requires_grad = False
    
    def init_weights(self):
        # extract weights from path file
        checkpoint="pretrain/mobile_sam.pt"
        state_dict_full = torch.load(checkpoint)
        prefix = "image_encoder."
        prefix_len = len(prefix)
        state_dict_vit = { k[prefix_len:] : v for k, v in state_dict_full.items() if k[:prefix_len] == prefix }
        self.load_state_dict(state_dict=state_dict_vit)
    
    def forward(self, x):
        x = self.forward_features(x)
        #x = self.norm_head(x)
        #x = self.head(x)
        return (x,)