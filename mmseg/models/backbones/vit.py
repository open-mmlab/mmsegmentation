import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule

from ..builder import BACKBONES
from ..utils import to_2tuple
from .base_backbone import BaseBackbone


@TRANSFORMER_LAYER.register_module()
class VitTransformerEncoderLayer(BaseTransformerLayer):
    """Implements encoder layer in Vit transformer."""

    def __init__(self, *args, **kwargs):
        super(VitTransformerEncoderLayer, self).__init__(*args, **kwargs)
        assert len(self.operation_order) == 4
        assert set(self.operation_order) == set(['self_attn', 'norm', 'ffn'])

    def init_weights(self):
        super(VitTransformerEncoderLayer, self).init_weights()
        for ffn in self.ffns:
            for m in ffn.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.normal_(m.bias, std=1e-6)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class VitTransformerEncoder(TransformerLayerSequence):
    """TransformerEncoder of Vit.

    Args:
        final_norm (dict): Whether to add a additional layer to normalize
            final feature map.
        out_indices (list | tuple | int): Output from which stages.
        coder_norm_cfg (dict): Config of last normalization layer. Only
            used when `self.pre_norm` and `self.final_norm` is `True`.
            Default: dict(type='LN')
    """

    def __init__(
            self,
            *args,
            final_norm,
            out_indices,
            coder_norm_cfg=dict(type='LN'),
            **kwargs,
    ):
        super(VitTransformerEncoder, self).__init__(*args, **kwargs)

        if isinstance(out_indices, int):
            self.out_indices = [
                out_indices,
            ]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')

        if coder_norm_cfg is not None:
            self.coder_norm = build_norm_layer(
                coder_norm_cfg, self.embed_dims)[1] if final_norm else None
        else:
            assert not final_norm, f'Use finalnorm in ' \
                                      f'{self.__class__.__name__},' \
                                      f'Please specify coder_norm_cfg'
            self.coder_norm = None

    def forward(self, query, key, value, *args, **kwargs):
        """Forward function for `TransformerCoder`.

        Returns:
            query_list: Output a list of feature vector and their indices
            refer to out_indices attribute. Each vector has shape
            [bs, num_query, embed_dims].
        """
        query_list = []
        for i, layer in enumerate(self.layers):
            query = layer(query=query, key=key, value=value, *args, **kwargs)
            if i == len(self.layers) - 1:
                if self.coder_norm is not None:
                    query = self.coder_norm(query)
            if i in self.out_indices:
                query_list.append(query)

        return query_list


# Modified from pytorch-image-models
class PatchEmbed(BaseModule):
    """Image to Patch Embedding.

    Args:
        img_size (int | tuple): The size of input image.
        patch_size (int): The size of one patch
        in_channels (int): The num of input channels.
        embed_dim (int): The dimensions of embedding.
        norm_cfg (dict, optional): Config dict for normalization layer.
        conv_cfg (dict, optional): The config dict for conv layers.
            Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 norm_cfg=None,
                 conv_cfg=None,
                 init_cfg=None):
        super(PatchEmbed, self).__init__(init_cfg)

        self.img_size = img_size
        self.patch_size = to_2tuple(patch_size)

        patches_resolution = [
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1]
        ]
        num_patches = patches_resolution[0] * patches_resolution[1]
        assert num_patches * self.patch_size[0] * self.patch_size[1] == \
               self.img_size[0] * self.img_size[1], \
               'The image size H*W must be divisible by patch size'
        self.patches_resolution = patches_resolution
        self.num_patches = num_patches

        # Use conv layer to embed
        self.projection = build_conv_layer(
            conv_cfg,
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't " \
            f'match model ({self.img_size[0]}*{self.img_size[1]}).'
        # The output size is (B, N, D), where N=H*W/P/P, D is embid_dim
        x = self.projection(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)

        return x


# Modified from pytorch-image-models
class HybridEmbed(BaseModule):
    """CNN Feature Map Embedding.

    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self,
                 backbone,
                 img_size=224,
                 feature_size=None,
                 in_channels=3,
                 embed_dim=768,
                 conv_cfg=None,
                 init_cfg=None):
        super(HybridEmbed, self).__init__(init_cfg)
        assert isinstance(backbone, nn.Module)

        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of
                #  determining the exact dim of the output feature
                #  map for all networks, the feature metadata has
                #  reliable channel and stride info, but using
                #  stride to calc feature dim requires info about padding of
                #  each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(
                    torch.zeros(1, in_channels, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    # last feature if backbone outputs list/tuple of features
                    o = o[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]

        # Use conv layer to embed
        self.projection = build_conv_layer(
            conv_cfg, feature_dim, embed_dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            # last feature if backbone outputs list/tuple of features
            x = x[-1]
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x


@BACKBONES.register_module()
class VisionTransformer(BaseBackbone):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words:
    Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    Args:
        embed_dim (int): Embedding dimension
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
        in_channels (int): Number of input channels
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0.
        hybrid_backbone (nn.Module, optional): CNN backbone to use in-place of
            PatchEmbed module. Default None.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Default: bicubic.
        with_cls_token (bool): If concatenating class token into image tokens
            as transformer input. Default: True.
        out_shape (str): Select the output format of feature information.
            Default: NCHW.
        encoder (`mmcv.ConfigDict` | Dict): Config of TransformerEncoder
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 drop_rate=0.,
                 hybrid_backbone=None,
                 interpolate_mode='bicubic',
                 with_cls_token=True,
                 out_shape='NCHW',
                 encoder=dict(
                     type='VitTransformerEncoder',
                     transformerlayers=None,
                     num_layers=12,
                     final_norm=False,
                     out_indices=None,
                     coder_norm_cfg=None,
                 ),
                 init_cfg=None):
        super(VisionTransformer, self).__init__(init_cfg)

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.interpolate_mode = interpolate_mode
        self.with_cls_token = with_cls_token
        self.out_shape = out_shape

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone,
                img_size=img_size,
                in_channels=in_channels,
                embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.encoder = build_transformer_layer_sequence(encoder)

    def init_weights(self, pretrained=None):
        super(VisionTransformer, self).init_weights(pretrained)

        if pretrained is None:
            # Modified from ClassyVision
            nn.init.normal_(self.pos_embed, std=0.02)

    def _pos_embeding(self, img, patched_img, pos_embed):
        """Positiong embeding method.

        Resize the pos_embed, if the input image size doesn't match
            the training size.
        Args:
            img (torch.Tensor): The inference image tensor, the shape
                must be [B, C, H, W].
            patched_img (torch.Tensor): The patched image, it should be
                shape of [B, L1, C].
            pos_embed (torch.Tensor): The pos_embed weighs, it should be
                shape of [B, L2, c].
        Return:
            torch.Tensor: The pos encoded image feature.
        """
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
            'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if pos_len == (self.img_size[0] // self.patch_size) * (
                    self.img_size[1] // self.patch_size) + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(
                    'Unexpected shape of pos_embed, got {}.'.format(
                        pos_embed.shape))
            pos_embed = self.resize_pos_embed(pos_embed, img.shape[2:],
                                              (pos_h, pos_w), self.patch_size,
                                              self.interpolate_mode)
        return self.pos_drop(patched_img + pos_embed)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, patch_size, mode):
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): pos_embed weights.
            input_shpae (tuple): Tuple for (input_h, intput_w).
            pos_shape (tuple): Tuple for (pos_h, pos_w).
            patch_size (int): Patch size.
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        input_h, input_w = input_shpae
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(
            pos_embed_weight,
            size=[input_h // patch_size, input_w // patch_size],
            align_corners=False,
            mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def forward(self, inputs):
        B = inputs.shape[0]
        x = self.patch_embed(inputs)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(inputs, x, self.pos_embed)

        if not self.with_cls_token:
            # Remove class token for transformer input
            x = x[:, 1:]

        query_list = self.encoder(query=x, key=None, value=None)

        outs = []
        for query in query_list:
            if self.with_cls_token:
                # Remove class token and reshape token for decoder head
                out = query[:, 1:]
            else:
                out = query
            if self.out_shape == 'NCHW':
                B, _, C = out.shape
                out = out.reshape(B, inputs.shape[2] // self.patch_size,
                                  inputs.shape[3] // self.patch_size,
                                  C).permute(0, 3, 1, 2)
            outs.append(out)
        return outs
