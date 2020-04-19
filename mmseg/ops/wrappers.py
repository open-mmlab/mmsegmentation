import warnings

import torch.nn.functional as F


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None:
            input_h, input_w = input.shape[2:]
            out_h, out_w = size
            if input_h % 2 and input_w % 2 and out_h % 2 and out_w % 2:
                if align_corners is False:
                    warnings.warn(
                        'When align_corners=False, the output '
                        'would more aligned if input/out size is `2x`')
            else:
                if align_corners is True:
                    warnings.warn(
                        'When align_corners=True, the output '
                        'would more aligned if input size {} and out '
                        'size {} is `2x+1`'.format((input_h, input_w),
                                                   (out_h, out_w)))
    return F.interpolate(input, size, scale_factor, mode, align_corners)
