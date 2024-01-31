/*!
**************************************************************************************************
* InternImage
* Copyright (c) 2022 OpenGVLab
* Licensed under The MIT License [see LICENSE for details]
**************************************************************************************************
* Modified from
*https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#pragma once
#include <torch/extension.h>

at::Tensor dcnv3_cpu_forward(const at::Tensor &input, const at::Tensor &offset,
                             const at::Tensor &mask, const int kernel_h,
                             const int kernel_w, const int stride_h,
                             const int stride_w, const int pad_h,
                             const int pad_w, const int dilation_h,
                             const int dilation_w, const int group,
                             const int group_channels, const float offset_scale,
                             const int im2col_step);

std::vector<at::Tensor>
dcnv3_cpu_backward(const at::Tensor &input, const at::Tensor &offset,
                   const at::Tensor &mask, const int kernel_h,
                   const int kernel_w, const int stride_h, const int stride_w,
                   const int pad_h, const int pad_w, const int dilation_h,
                   const int dilation_w, const int group,
                   const int group_channels, const float offset_scale,
                   const at::Tensor &grad_output, const int im2col_step);
