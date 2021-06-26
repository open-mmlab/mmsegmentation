import torch
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner.base_module import BaseModule
from torch.nn.modules.utils import _pair as to_2tuple


class PatchEmbed(BaseModule):
    """Image to Patch Embedding V2.

    We use a conv layer to implement PatchEmbed.
    Args:
        img_size (int | tuple): The size of input image. Default: 224
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (dict, optional): The config dict for conv layers type
            selection. Default: None.
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: None (Default to be equal with kernel_size).
        padding (int): The padding length of embedding conv. Default: 0.
        dilation (int): The dilation rate of embedding conv. Default: 1.
        norm_cfg (dict, optional): Config dict for normalization layer.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 embed_dims=768,
                 conv_type=None,
                 kernel_size=16,
                 stride=16,
                 padding=0,
                 dilation=1,
                 norm_cfg=None,
                 init_cfg=None):
        super(PatchEmbed, self).__init__(init_cfg)

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        self.img_size = img_size
        self.embed_dims = embed_dims

        if stride is None:
            stride = kernel_size

        # Use conv layer to embed
        conv_type = conv_type or dict(type='Conv2d')
        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation)

        with torch.no_grad():
            # FIXME this is hacky, but the most simple way to calculate
            #  how many patches a input image is splited to. As the padding
            #  attribute in conv_cfg can be a string such as "same",
            #  it is complicated to calculate by formula
            o = self.projection(
                torch.zeros(1, in_channels, self.img_size[0],
                            self.img_size[1]))
            patches_resolution = o.shape[-2:]
            num_patches = patches_resolution[0] * patches_resolution[1]

        self.patches_resolution = patches_resolution
        self.num_patches = num_patches

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

    def forward(self, x):
        x = self.projection(x).flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x
