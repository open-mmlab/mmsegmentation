# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
import math
from torch.autograd import gradcheck

from functions.dcnv3_func import DCNv3Function, dcnv3_core_pytorch

H_in, W_in = 8, 8
N, M, D = 2, 4, 16
Kh, Kw = 3, 3
P = Kh * Kw
offset_scale = 2.0
pad = 1
dilation = 1
stride = 1
H_out = (H_in + 2 * pad - (dilation * (Kh - 1) + 1)) // stride + 1
W_out = (W_in + 2 * pad - (dilation * (Kw - 1) + 1)) // stride + 1

torch.manual_seed(3)


@torch.no_grad()
def check_forward_equal_with_pytorch_double():
    input = torch.rand(N, H_in, W_in, M*D).cuda() * 0.01
    offset = torch.rand(N, H_out, W_out, M*P*2).cuda() * 10
    mask = torch.rand(N, H_out, W_out, M, P).cuda() + 1e-5
    mask /= mask.sum(-1, keepdim=True)
    mask = mask.reshape(N, H_out, W_out, M*P)

    output_pytorch = dcnv3_core_pytorch(
        input.double(),
        offset.double(),
        mask.double(),
        Kh, Kw, stride, stride, Kh // 2, Kw // 2, dilation, dilation, M, D, offset_scale).detach().cpu()

    im2col_step = 2
    output_cuda = DCNv3Function.apply(
        input.double(),
        offset.double(),
        mask.double(),
        Kh, Kw, stride, stride, Kh // 2, Kw // 2, dilation, dilation, M, D, offset_scale,
        im2col_step).detach().cpu()

    fwdok = torch.allclose(output_cuda, output_pytorch)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() /
                   output_pytorch.abs()).max()
    print('>>> forward double')
    print(f'* {fwdok} check_forward_equal_with_pytorch_double: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')


@torch.no_grad()
def check_forward_equal_with_pytorch_float():
    input = torch.rand(N, H_in, W_in, M*D).cuda() * 0.01
    offset = torch.rand(N, H_out, W_out, M*P*2).cuda() * 10
    mask = torch.rand(N, H_out, W_out, M, P).cuda() + 1e-5
    mask /= mask.sum(-1, keepdim=True)
    mask = mask.reshape(N, H_out, W_out, M*P)

    output_pytorch = dcnv3_core_pytorch(
        input,
        offset,
        mask,
        Kh, Kw, stride, stride, Kh // 2, Kw // 2, dilation, dilation, M, D, offset_scale).detach().cpu()

    im2col_step = 2
    output_cuda = DCNv3Function.apply(
        input,
        offset,
        mask,
        Kh, Kw, stride, stride, Kh // 2, Kw // 2, dilation, dilation, M, D, offset_scale,
        im2col_step).detach().cpu()

    fwdok = torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() /
                   output_pytorch.abs()).max()
    print('>>> forward float')
    print(f'* {fwdok} check_forward_equal_with_pytorch_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')


def check_backward_equal_with_pytorch_double(channels=4, grad_input=True, grad_offset=True, grad_mask=True):
    # H_in, W_in = 4, 4
    N = 2
    M = 2
    H_out = (H_in + 2 * pad - (dilation * (Kh - 1) + 1)) // stride + 1
    W_out = (W_in + 2 * pad - (dilation * (Kw - 1) + 1)) // stride + 1

    D = channels
    input0 = torch.rand(N, H_in, W_in, M*D).cuda() * 0.01
    offset0 = torch.rand(N, H_out, W_out, M*P*2).cuda() * 10
    mask0 = torch.rand(N, H_out, W_out, M, P).cuda() + 1e-5
    mask0 /= mask0.sum(-1, keepdim=True)
    mask0 = mask0.reshape(N, H_out, W_out, M*P)
    input0.requires_grad = grad_input
    offset0.requires_grad = grad_offset
    mask0.requires_grad = grad_mask

    output_pytorch = dcnv3_core_pytorch(
        input0.double(),
        offset0.double(),
        mask0.double(),
        Kh, Kw, stride, stride, Kh // 2, Kw // 2, dilation, dilation, M, D, offset_scale)
    output_pytorch.sum().backward()

    input1 = input0.detach()
    offset1 = offset0.detach()
    mask1 = mask0.detach()
    input1.requires_grad = grad_input
    offset1.requires_grad = grad_offset
    mask1.requires_grad = grad_mask

    im2col_step = 2
    output_cuda = DCNv3Function.apply(
        input1.double(),
        offset1.double(),
        mask1.double(),
        Kh, Kw, stride, stride, Kh // 2, Kw // 2, dilation, dilation, M, D, offset_scale,
        im2col_step)
    output_cuda.sum().backward()

    print(f'>>> backward double: channels {D}')
    bwdok = torch.allclose(input0.grad, input1.grad, rtol=1e-2, atol=1e-3)
    max_abs_err = (input0.grad - input1.grad).abs().max()
    max_rel_err = ((input0.grad - input1.grad).abs() /
                   input0.grad.abs()).max()
    print(
        f'* {bwdok} input_grad check_backward_equal_with_pytorch_double: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')

    bwdok = torch.allclose(offset0.grad, offset1.grad, rtol=1e-2, atol=1e-3)
    max_abs_err = (offset0.grad - offset1.grad).abs().max()
    max_rel_err = ((offset0.grad - offset1.grad).abs() /
                   offset0.grad.abs()).max()
    print(
        f'* {bwdok} offset_grad check_backward_equal_with_pytorch_double: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')

    bwdok = torch.allclose(mask0.grad, mask1.grad, rtol=1e-2, atol=1e-3)
    max_abs_err = (mask0.grad - mask1.grad).abs().max()
    max_rel_err = ((mask0.grad - mask1.grad).abs() /
                   mask0.grad.abs()).max()
    print(
        f'* {bwdok} mask_grad check_backward_equal_with_pytorch_double: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')


def check_backward_equal_with_pytorch_float(channels=4, grad_input=True, grad_offset=True, grad_mask=True):
    # H_in, W_in = 4, 4
    N = 2
    M = 2
    H_out = (H_in + 2 * pad - (dilation * (Kh - 1) + 1)) // stride + 1
    W_out = (W_in + 2 * pad - (dilation * (Kw - 1) + 1)) // stride + 1

    D = channels
    input0 = torch.rand(N, H_in, W_in, M*D).cuda() * 0.01
    offset0 = torch.rand(N, H_out, W_out, M*P*2).cuda() * 10
    mask0 = torch.rand(N, H_out, W_out, M, P).cuda() + 1e-5
    mask0 /= mask0.sum(-1, keepdim=True)
    mask0 = mask0.reshape(N, H_out, W_out, M*P)
    input0.requires_grad = grad_input
    offset0.requires_grad = grad_offset
    mask0.requires_grad = grad_mask

    output_pytorch = dcnv3_core_pytorch(
        input0,
        offset0,
        mask0,
        Kh, Kw, stride, stride, Kh // 2, Kw // 2, dilation, dilation, M, D, offset_scale)
    output_pytorch.sum().backward()

    input1 = input0.detach()
    offset1 = offset0.detach()
    mask1 = mask0.detach()
    input1.requires_grad = grad_input
    offset1.requires_grad = grad_offset
    mask1.requires_grad = grad_mask

    im2col_step = 2
    output_cuda = DCNv3Function.apply(
        input1,
        offset1,
        mask1,
        Kh, Kw, stride, stride, Kh // 2, Kw // 2, dilation, dilation, M, D, offset_scale,
        im2col_step)
    output_cuda.sum().backward()

    print(f'>>> backward float: channels {D}')
    bwdok = torch.allclose(input0.grad, input1.grad, rtol=1e-2, atol=1e-3)
    max_abs_err = (input0.grad - input1.grad).abs().max()
    max_rel_err = ((input0.grad - input1.grad).abs() /
                   input0.grad.abs()).max()
    print(
        f'* {bwdok} input_grad check_backward_equal_with_pytorch_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')

    bwdok = torch.allclose(offset0.grad, offset1.grad, rtol=1e-2, atol=1e-3)
    max_abs_err = (offset0.grad - offset1.grad).abs().max()
    max_rel_err = ((offset0.grad - offset1.grad).abs() /
                   offset0.grad.abs()).max()
    print(
        f'* {bwdok} offset_grad check_backward_equal_with_pytorch_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')

    bwdok = torch.allclose(mask0.grad, mask1.grad, rtol=1e-2, atol=1e-3)
    max_abs_err = (mask0.grad - mask1.grad).abs().max()
    max_rel_err = ((mask0.grad - mask1.grad).abs() /
                   mask0.grad.abs()).max()
    print(
        f'* {bwdok} mask_grad check_backward_equal_with_pytorch_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')


@torch.no_grad()
def check_time_cost(im2col_step=128):
    N = 512
    H_in, W_in = 64, 64
    H_out = (H_in + 2 * pad - (dilation * (Kh - 1) + 1)) // stride + 1
    W_out = (W_in + 2 * pad - (dilation * (Kw - 1) + 1)) // stride + 1

    input = torch.rand(N, H_in, W_in, M*D).cuda() * 0.01
    offset = torch.rand(N, H_out, W_out, M*P*2).cuda() * 10
    mask = torch.rand(N, H_out, W_out, M, P).cuda() + 1e-5
    mask /= mask.sum(-1, keepdim=True)
    mask = mask.reshape(N, H_out, W_out, M*P)
    print(
        f'>>> time cost: im2col_step {im2col_step}; input {input.shape}; points {P} ')
    repeat = 100
    for i in range(repeat):
        output_cuda = DCNv3Function.apply(
            input,
            offset,
            mask,
            Kh, Kw, stride, stride, Kh // 2, Kw // 2, dilation, dilation, M, D, 1.0,
            im2col_step)
    torch.cuda.synchronize()
    start = time.time()
    for i in range(repeat):
        output_cuda = DCNv3Function.apply(
            input,
            offset,
            mask,
            Kh, Kw, stride, stride, Kh // 2, Kw // 2, dilation, dilation, M, D, 1.0,
            im2col_step)
    torch.cuda.synchronize()
    print(f'foward time cost: {(time.time() - start) / repeat}')


if __name__ == '__main__':
    check_forward_equal_with_pytorch_double()
    check_forward_equal_with_pytorch_float()
    for channels in [1, 16, 30, 32, 64, 71, 1025]:
        check_backward_equal_with_pytorch_double(channels, True, True, True)
    for channels in [1, 16, 30, 32, 64, 71, 1025]:
        check_backward_equal_with_pytorch_float(channels, True, True, True)
    for i in range(3):
        im2col_step = 128 * (2 ** i)
        check_time_cost(im2col_step)
