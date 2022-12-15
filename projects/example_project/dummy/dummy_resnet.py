from mmseg.models.backbones import ResNetV1c
from mmseg.registry import MODELS


@MODELS.register_module()
class DummyResNet(ResNetV1c):
    """Implements a dummy ResNet wrapper for demonstration purpose.
    Args:
        **kwargs: All the arguments are passed to the parent class.
    """

    def __init__(self, **kwargs) -> None:
        print('Hello world!')
        super().__init__(**kwargs)
