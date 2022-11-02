# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np
import pytest
from mmengine import FileClient

from mmseg.utils import datafrombytes


@pytest.mark.parametrize(
    ['backend', 'suffix'],
    [['nifti', '.nii.gz'], ['numpy', '.npy'], ['pickle', '.pkl']])
def test_datafrombytes(backend, suffix):

    file_client = FileClient('disk')
    file_path = osp.join(osp.dirname(__file__), '../data/biomedical' + suffix)
    bytes = file_client.get(file_path)
    data = datafrombytes(bytes, backend)

    if backend == 'pickle':
        # test pickle loading
        assert isinstance(data, dict)
    else:
        assert isinstance(data, np.ndarray)
        if backend == 'nifti':
            # test nifti file loading
            assert len(data.shape) == 3
        else:
            # test npy file loading
            # testing data biomedical.npy includes data and label
            assert len(data.shape) == 4
            assert data.shape[0] == 2
