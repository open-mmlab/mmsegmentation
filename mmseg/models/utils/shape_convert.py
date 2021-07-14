def nlc_to_nchw(tensor, hw_shape):
    H, W = hw_shape
    assert len(tensor.shape) == 3
    B, L, C = tensor.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    return tensor.transpose(1, 2).reshape(B, C, H, W)


def nchw_to_nlc(tensor):
    assert len(tensor.shape) == 4
    return tensor.flatten(2).transpose(1, 2).contiguous()
