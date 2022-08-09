# Copyright (c) OpenMMLab. All rights reserved.
from .accuracy import Accuracy, accuracy
from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy, mask_cross_entropy)
from .dice_loss import DiceLoss
from .focal_loss import FocalLoss
from .lovasz_loss import LovaszLoss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss
from .dummy_loss import DummyLoss, dummy_loss
from .belief_matching_loss import BeliefMatchingLoss
from .ldam_loss import LDAMLoss
from .edl_loss import EDLLoss
from .mse_loss import MSELoss
from .balanced_softmax_loss import BalancedSoftmaxLoss
__all__ = [
    'accuracy', 'Accuracy', 'cross_entropy', 'binary_cross_entropy',
    'mask_cross_entropy', 'CrossEntropyLoss', 'reduce_loss',
    'weight_reduce_loss', 'weighted_loss', 'LovaszLoss', 'DiceLoss',
    'FocalLoss', 'LDAMLoss', 'EDLLoss', 'BeliefMatchingLoss', 'MSELoss', 'BalancedSoftmaxLoss'
]
