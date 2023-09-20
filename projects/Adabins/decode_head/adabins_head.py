from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from torch import Tensor

from mmseg.registry import MODELS


class PatchTransformerEncoder(nn.Module):
    """the Patch Transformer Encoder.

    Args:
        in_channels (int): the channels of input
        patch_size (int): the path size
        embedding_dim (int): The feature dimension.
        num_heads (int): the number of encoder head
        conv_cfg (dict): Config dict for convolution layer.
    """

    def __init__(self,
                 in_channels,
                 patch_size=10,
                 embedding_dim=128,
                 num_heads=4,
                 conv_cfg=dict(type='Conv')):
        super().__init__()
        encoder_layers = nn.TransformerEncoderLayer(
            embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=4)  # takes shape S,N,E

        self.embedding_convPxP = build_conv_layer(
            conv_cfg,
            in_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size)
        self.positional_encodings = nn.Parameter(
            torch.rand(500, embedding_dim), requires_grad=True)

    def forward(self, x):
        embeddings = self.embedding_convPxP(x).flatten(
            2)  # .shape = n,c,s = n, embedding_dim, s
        embeddings = embeddings + self.positional_encodings[:embeddings.shape[
            2], :].T.unsqueeze(0)

        # change to S,N,E format required by transformer
        embeddings = embeddings.permute(2, 0, 1)
        x = self.transformer_encoder(embeddings)  # .shape = S, N, E
        return x


class PixelWiseDotProduct(nn.Module):
    """the pixel wise dot product."""

    def __init__(self):
        super().__init__()

    def forward(self, x, K):
        n, c, h, w = x.size()
        _, cout, ck = K.size()
        assert c == ck, 'Number of channels in x and Embedding dimension ' \
                        '(at dim 2) of K matrix must match'
        y = torch.matmul(
            x.view(n, c, h * w).permute(0, 2, 1),
            K.permute(0, 2, 1))  # .shape = n, hw, cout
        return y.permute(0, 2, 1).view(n, cout, h, w)


@MODELS.register_module()
class AdabinsHead(nn.Module):
    """the head of the adabins,include mViT.

    Args:
        in_channels (int):the channels of the input
        n_query_channels (int):the channels of the query
        patch_size (int): the patch size
        embedding_dim (int):The feature dimension.
        num_heads (int):the number of head
        n_bins (int):the number of bins
        min_val (float): the min width of bin
        max_val (float): the max width of bin
        conv_cfg (dict): Config dict for convolution layer.
        norm (str): the activate method
        align_corners (bool, optional): Geometrically, we consider the pixels
            of the input and output as squares rather than points.
    """

    def __init__(self,
                 in_channels,
                 n_query_channels=128,
                 patch_size=16,
                 embedding_dim=128,
                 num_heads=4,
                 n_bins=100,
                 min_val=0.1,
                 max_val=10,
                 conv_cfg=dict(type='Conv'),
                 norm='linear',
                 align_corners=False,
                 threshold=0):
        super().__init__()
        self.out_channels = n_bins
        self.align_corners = align_corners
        self.norm = norm
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.n_query_channels = n_query_channels
        self.patch_transformer = PatchTransformerEncoder(
            in_channels, patch_size, embedding_dim, num_heads)
        self.dot_product_layer = PixelWiseDotProduct()
        self.threshold = threshold
        self.conv3x3 = build_conv_layer(
            conv_cfg,
            in_channels,
            embedding_dim,
            kernel_size=3,
            stride=1,
            padding=1)
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, 256), nn.LeakyReLU(), nn.Linear(256, 256),
            nn.LeakyReLU(), nn.Linear(256, n_bins))
        self.conv_out = nn.Sequential(
            build_conv_layer(conv_cfg, in_channels, n_bins, kernel_size=1),
            nn.Softmax(dim=1))

    def forward(self, x):
        # n, c, h, w = x.size()
        tgt = self.patch_transformer(x.clone())  # .shape = S, N, E

        x = self.conv3x3(x)

        regression_head, queries = tgt[0,
                                       ...], tgt[1:self.n_query_channels + 1,
                                                 ...]

        # Change from S, N, E to N, S, E
        queries = queries.permute(1, 0, 2)
        range_attention_maps = self.dot_product_layer(
            x, queries)  # .shape = n, n_query_channels, h, w

        y = self.regressor(regression_head)  # .shape = N, dim_out
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        bin_widths_normed = y / y.sum(dim=1, keepdim=True)
        out = self.conv_out(range_attention_maps)

        bin_widths = (self.max_val -
                      self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = F.pad(
            bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dim_out = centers.size()
        centers = centers.view(n, dim_out, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)
        return bin_edges, pred

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg, **kwargs) -> Tensor:
        """Forward function for testing, only ``pam_cam`` is used."""
        pred = self.forward(inputs)[-1]
        final = torch.clamp(pred, self.min_val, self.max_val)

        final[torch.isinf(final)] = self.max_val
        final[torch.isnan(final)] = self.min_val
        return final
