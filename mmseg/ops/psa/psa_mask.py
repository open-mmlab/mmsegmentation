# Modified from https://github.com/hszhao/semseg/blob/master/lib/psa
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from . import psamask_ext


class PSAMaskFunction(Function):

    @staticmethod
    def forward(ctx, input, psa_type, mask_size):
        ctx.psa_type = psa_type
        ctx.mask_size = _pair(mask_size)
        ctx.save_for_backward(input)

        h_mask, w_mask = ctx.mask_size
        batch_size, channels, h_feature, w_feature = input.size()
        assert channels == h_mask * w_mask
        output = input.new_zeros(
            (batch_size, h_feature * w_feature, h_feature, w_feature))
        psamask_ext.psamask_forward(psa_type, input, output, batch_size,
                                    h_feature, w_feature, h_mask, w_mask,
                                    (h_mask - 1) // 2, (w_mask - 1) // 2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        psa_type = ctx.psa_type
        h_mask, w_mask = ctx.mask_size
        batch_size, channels, h_feature, w_feature = input.size()
        grad_input = grad_output.new_zeros(
            (batch_size, channels, h_feature, w_feature))
        psamask_ext.psamask_backward(psa_type, grad_output, grad_input,
                                     batch_size, h_feature, w_feature, h_mask,
                                     w_mask, (h_mask - 1) // 2,
                                     (w_mask - 1) // 2)
        return grad_input, None, None, None


psa_mask = PSAMaskFunction.apply


class PSAMask(nn.Module):

    def __init__(self, psa_type, mask_size=None):
        super(PSAMask, self).__init__()
        assert psa_type in ['collect', 'distribute']
        if psa_type == 'collect':
            psa_type_enum = 0
        else:
            psa_type_enum = 1
        self.psa_type_enum = psa_type_enum
        self.mask_size = mask_size

    def forward(self, input):
        return psa_mask(input, self.psa_type_enum, self.mask_size)
