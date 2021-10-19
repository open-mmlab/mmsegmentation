# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa
import warnings

from .formatting import *

warnings.warn('DeprecationWarning: mmseg.datasets.pipelines.formating will be '
              'deprecated in 2021, please replace it with '
              'mmseg.datasets.pipelines.formatting.')
