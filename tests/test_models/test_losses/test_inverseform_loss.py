# Copyright (c) OpenMMLab. All rights reserved.
import urllib

import pytest
import torch

from mmseg.models.losses import InverseFormLoss


def test_inverseform_loss():
    pred = torch.zeros(3, 1, 4, 4)
    target = torch.zeros(3, 1, 4, 4)

    for i in range(3):
        pred[i, 0, i, :] = 1
        target[i, 0, :, i] = 1
        print(f'pred is:{pred[i]}, \n target is {target[i]}')

    # test inverseform_loss
    pretraind_model_url = 'https://github.com/Qualcomm-AI-research/InverseForm\
        /releases/download/v1.0/distance_measures_regressor.pth'

    inverseNet_path = './checkpoints/distance_measures_regressor.pth'
    urllib.request.urlretrieve(pretraind_model_url, inverseNet_path)

    loss_class = InverseFormLoss(inverseNet_path=inverseNet_path)
    with pytest.raises(AssertionError):
        loss = loss_class(pred, target)
    assert isinstance(loss, torch.Tensor)


# if __name__ == "__main__":
#     test_inverseform_loss()
