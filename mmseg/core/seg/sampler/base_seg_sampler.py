from abc import ABCMeta, abstractmethod


class BasSegSampler(metaclass=ABCMeta):

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def sample(self, seg_logit, seg_label):
        pass
