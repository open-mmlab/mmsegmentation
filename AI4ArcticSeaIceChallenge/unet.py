#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""U-Net model."""

# -- File info -- #
__author__ = 'Andreas R. Stokholm'
__contributor__ = 'Andrzej S. Kucik'
__copyright__ = ['Technical University of Denmark', 'European Space Agency']
__contact__ = ['stokholm@space.dtu.dk', 'andrzej.kucik@esa.int']
__version__ = '0.3.0'
__date__ = '2022-09-20'


# -- Third-party modules -- #
import torch


class UNet(torch.nn.Module):
    """PyTorch U-Net Class. Uses unet_parts."""

    def __init__(self, options):
        super().__init__()

        self.input_block = DoubleConv(options, input_n=len(options['train_variables']), output_n=options['unet_conv_filters'][0])

        self.contract_blocks = torch.nn.ModuleList()
        for contract_n in range(1, len(options['unet_conv_filters'])):
            self.contract_blocks.append(
                ContractingBlock(options=options,
                                 input_n=options['unet_conv_filters'][contract_n - 1],
                                 output_n=options['unet_conv_filters'][contract_n]))  # only used to contract input patch.

        self.bridge = ContractingBlock(options, input_n=options['unet_conv_filters'][-1], output_n=options['unet_conv_filters'][-1])

        self.expand_blocks = torch.nn.ModuleList()
        self.expand_blocks.append(
            ExpandingBlock(options=options, input_n=options['unet_conv_filters'][-1], output_n=options['unet_conv_filters'][-1]))

        for expand_n in range(len(options['unet_conv_filters']), 1, -1):
            self.expand_blocks.append(ExpandingBlock(options=options,
                                                     input_n=options['unet_conv_filters'][expand_n - 1],
                                                     output_n=options['unet_conv_filters'][expand_n - 2]))

        self.sic_feature_map = FeatureMap(input_n=options['unet_conv_filters'][0], output_n=options['n_classes']['SIC'])
        self.sod_feature_map = FeatureMap(input_n=options['unet_conv_filters'][0], output_n=options['n_classes']['SOD'])
        self.floe_feature_map = FeatureMap(input_n=options['unet_conv_filters'][0], output_n=options['n_classes']['FLOE'])

    def forward(self, x):
        """Forward model pass."""
        x_contract = [self.input_block(x)]
        for contract_block in self.contract_blocks:
            x_contract.append(contract_block(x_contract[-1]))
        x_expand = self.bridge(x_contract[-1])
        up_idx = len(x_contract)
        for expand_block in self.expand_blocks:
            x_expand = expand_block(x_expand, x_contract[up_idx - 1])
            up_idx -= 1            

        return {'SIC': self.sic_feature_map(x_expand),
                'SOD': self.sod_feature_map(x_expand),
                'FLOE': self.floe_feature_map(x_expand)}


class FeatureMap(torch.nn.Module):
    """Class to perform final 1D convolution before calculating cross entropy or using softmax."""

    def __init__(self, input_n, output_n):
        super(FeatureMap, self).__init__()

        self.feature_out = torch.nn.Conv2d(input_n, output_n, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        """Pass x through final layer."""
        return self.feature_out(x)


class DoubleConv(torch.nn.Module):
    """Class to perform a double conv layer in the U-NET architecture. Used in unet_model.py."""

    def __init__(self, options, input_n, output_n):
        super(DoubleConv, self).__init__()

        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_n,
                      out_channels=output_n,
                      kernel_size=options['conv_kernel_size'],
                      stride=options['conv_stride_rate'],
                      padding=options['conv_padding'],
                      padding_mode=options['conv_padding_style'],
                      bias=False),
            torch.nn.BatchNorm2d(output_n),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=output_n,
                      out_channels=output_n,
                      kernel_size=options['conv_kernel_size'],
                      stride=options['conv_stride_rate'],
                      padding=options['conv_padding'],
                      padding_mode=options['conv_padding_style'],
                      bias=False),
            torch.nn.BatchNorm2d(output_n),
            torch.nn.ReLU()
        )

    def forward(self, x):
        """Pass x through the double conv layer."""
        x = self.double_conv(x)

        return x


class ContractingBlock(torch.nn.Module):
    """Class to perform downward pass in the U-Net."""

    def __init__(self, options, input_n, output_n):
        super(ContractingBlock, self).__init__()

        self.contract_block = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.double_conv = DoubleConv(options, input_n, output_n)

    def forward(self, x):
        """Pass x through the downward layer."""
        x = self.contract_block(x)
        x = self.double_conv(x)
        return x


class ExpandingBlock(torch.nn.Module):
    """Class to perform upward layer in the U-Net."""

    def __init__(self, options, input_n, output_n):
        super(ExpandingBlock, self).__init__()

        self.padding_style = options['conv_padding_style']
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.double_conv = DoubleConv(options, input_n=input_n + output_n, output_n=output_n)

    def forward(self, x, x_skip):
        """Pass x through the upward layer and concatenate with opposite layer."""
        x = self.upsample(x)

        # Insure that x and skip H and W dimensions match.
        x = expand_padding(x, x_skip, padding_style=self.padding_style)
        x = torch.cat([x, x_skip], dim=1)

        return self.double_conv(x)
    
    
def expand_padding(x, x_contract, padding_style: str = 'constant'):
    """
    Insure that x and x_skip H and W dimensions match.
    Parameters
    ----------
    x :
        Image tensor of shape (batch size, channels, height, width). Expanding path.
    x_contract :
        Image tensor of shape (batch size, channels, height, width) Contracting path.
        or torch.Size. Contracting path.
    padding_style : str
        Type of padding.

    Returns
    -------
    x : ndtensor
        Padded expanding path.
    """
    # Check whether x_contract is tensor or shape.
    if type(x_contract) == type(x):
        x_contract = x_contract.size()

    # Calculate necessary padding to retain patch size.
    pad_y = x_contract[2] - x.size()[2]
    pad_x = x_contract[3] - x.size()[3]

    if padding_style == 'zeros':
        padding_style = 'constant'

    x = torch.nn.functional.pad(x, [pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2], mode=padding_style)

    return x
