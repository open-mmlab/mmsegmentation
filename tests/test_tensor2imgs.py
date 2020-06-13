import mmcv
import numpy as np
import pytest
import torch

from mmseg.core import tensor2imgs


def test_tensor2imgs():
    img_tensor = torch.FloatTensor(2, 3, 4, 4).uniform_(0, 1)

    with pytest.raises(AssertionError):
        # input is not a tensor
        tensor2imgs(4)
    with pytest.raises(AssertionError):
        # unsupported 2D tensor
        tensor2imgs(torch.FloatTensor(2, 4).uniform_(0, 1))

    imgs = tensor2imgs(img_tensor)
    assert mmcv.is_list_of(imgs, np.ndarray)
    assert len(imgs) == img_tensor.shape[0]
