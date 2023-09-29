# Copyright (c) OpenMMLab. All rights reserved.

import os

import pytest
import requests
import torch
from tqdm import tqdm


def test_inverseform_loss():
    from mmseg.models.losses import InverseFormLoss

    # construct test data
    pred = torch.zeros(3, 1, 4, 4)
    target = torch.zeros(3, 1, 4, 4)

    for i in range(3):
        pred[i, 0, i, :] = 1
        target[i, 0, :, i] = 1
        print(f'pred is:{pred[i]}, \n target is {target[i]}')

    # download pretrained model
    repo = 'https://github.com/Qualcomm-AI-research/InverseForm/'
    download_folder = 'releases/download/v1.0/distance_measures_regressor.pth'
    pretraind_model_url = repo + download_folder
    print(pretraind_model_url)
    os.makedirs('./checkpoints', exist_ok=True)
    inverseNet_path = './checkpoints/distance_measures_regressor.pth'
    if not os.path.exists(inverseNet_path):
        response = requests.get(pretraind_model_url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(inverseNet_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print('ERROR, something went wrong')

    # test inverseform_loss
    loss_class = InverseFormLoss(
        inverseNet_path=inverseNet_path, map_location='cpu')
    loss = loss_class(pred, target)
    print(loss)
    assert isinstance(loss, torch.Tensor)

    # Test loss forward with pred.shape!=target.shape
    pred_wrong = torch.zeros(2, 1, 4, 4)
    target_wrong = torch.zeros(3, 1, 4, 4)
    with pytest.raises(AssertionError):
        loss = loss_class(pred_wrong, target_wrong)
        print(loss)
