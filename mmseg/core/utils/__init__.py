# Copyright (c) OpenMMLab. All rights reserved.
from .dist_util import check_dist_init, sync_random_seed
from .misc import add_prefix

__all__ = ['add_prefix', 'check_dist_init', 'sync_random_seed']
