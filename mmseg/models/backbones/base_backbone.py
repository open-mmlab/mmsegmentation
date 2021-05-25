import logging
from abc import ABCMeta, abstractmethod

from mmcv.runner import load_checkpoint
from mmcv.runner.base_module import BaseModule


class BaseBackbone(BaseModule, metaclass=ABCMeta):
    """Base backbone.

    This class defines the basic functions of a backbone. Any backbone that
    inherits this class should at least define its own `forward` function.
    """

    def __init__(self, init_cfg=None):
        super(BaseBackbone, self).__init__(init_cfg)

    def init_weights(self, pretrained=None):
        """Init backbone weights.

        Args:
            pretrained (str | None): If pretrained is a string, then it
                initializes backbone weights by loading the pretrained
                checkpoint. If pretrained is None, then it follows default
                initializer or customized initializer in subclasses.
        """
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            # use default initializer or customized initializer in subclasses
            super(BaseBackbone, self).init_weights()
        else:
            raise TypeError('pretrained must be a str or None.'
                            f' But received {type(pretrained)}.')

    @abstractmethod
    def forward(self, x):
        """Forward computation.

        Args:
            x (tensor | tuple[tensor]): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
        """
        pass

    def train(self, mode=True):
        """Set module status before forward computation.

        Args:
            mode (bool): Whether it is train_mode or test_mode
        """
        super(BaseBackbone, self).train(mode)
