# Copyright (c) OpenMMLab. All rights reserved.
import gzip
import io
import pickle

import numpy as np


def datafrombytes(content: bytes, backend: str = 'numpy') -> np.ndarray:
    """Data decoding from bytes.

    Args:
        content (bytes): The data bytes got from files or other streams.
        backend (str): The data decoding backend type. Options are 'numpy',
            'nifti' and 'pickle'. Defaults to 'numpy'.

    Returns:
        numpy.ndarray: Loaded data array.
    """
    if backend == 'pickle':
        data = pickle.loads(content)
    else:
        with io.BytesIO(content) as f:
            if backend == 'nifti':
                f = gzip.open(f)
                try:
                    from nibabel import FileHolder, Nifti1Image
                except ImportError:
                    print('nifti files io depends on nibabel, please run'
                          '`pip install nibabel` to install it')
                fh = FileHolder(fileobj=f)
                data = Nifti1Image.from_file_map({'header': fh, 'image': fh})
                data = Nifti1Image.from_bytes(data.to_bytes()).get_fdata()
            elif backend == 'numpy':
                data = np.load(f)
            else:
                raise ValueError
    return data
