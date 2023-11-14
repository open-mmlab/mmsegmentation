from typing import Tuple, Type, List, Union
from segment_anything.modeling.image_encoder import ImageEncoderViT
import torch
from torch.nn import functional as F
from functools import partial

from mmseg.registry import MODELS
from mmengine.model.base_module import BaseModule


@MODELS.register_module()
class SAMImageEncoderViT_B( BaseModule, ImageEncoderViT):

    def __init__(self, init_cfg: Union[dict, List[dict], None] = None) -> None:
        BaseModule.__init__(self, init_cfg)
        ImageEncoderViT.__init__(
            self,
            depth=12,
            embed_dim=768,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads = 12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2,5,8,11],
            window_size=14,
            out_chans=256,
        )

        # freeze all params
        for _, param in self.named_parameters():
            param.requires_grad = False
    
    def init_weights(self):
        # extract weights from path file
        checkpoint="/data/pretrain/sam_vit_b_01ec64.pth"
        state_dict_full = torch.load(checkpoint)
        prefix = "image_encoder."
        prefix_len = len(prefix)
        state_dict_vit = { k[prefix_len:] : v for k, v in state_dict_full.items() if k[:prefix_len] == prefix }
        self.load_state_dict(state_dict=state_dict_vit)
    
