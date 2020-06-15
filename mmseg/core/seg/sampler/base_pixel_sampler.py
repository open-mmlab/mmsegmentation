from abc import ABCMeta, abstractmethod


class BasePixelSampler(metaclass=ABCMeta):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def sample(self, seg_logit, seg_label):
        pass
