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

#include "dcnv3.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dcnv3_forward", &dcnv3_forward, "dcnv3_forward");
    m.def("dcnv3_backward", &dcnv3_backward, "dcnv3_backward");
}
