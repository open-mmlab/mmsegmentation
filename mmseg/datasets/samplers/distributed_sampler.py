# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division

import numpy as np
import torch
from torch.utils.data import DistributedSampler as _DistributedSampler

from mmseg.utils import sync_random_seed


class DistributedSampler(_DistributedSampler):
    """DistributedSampler inheriting from
    `torch.utils.data.DistributedSampler`.

    In pytorch of lower versions, there is no `shuffle` argument. This child
    class will port one to DistributedSampler.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 samples_per_gpu=1,
                 seed=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)

        self.shuffle = shuffle
        self.samples_per_gpu = samples_per_gpu
        # fix the bug of the official implementation
        self.num_samples_per_replica = int(
            int(
                np.ceil(
                    len(self.dataset) * 1.0 / self.num_replicas /
                    samples_per_gpu)))
        self.num_samples = self.num_samples_per_replica * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

        # to avoid padding bug when meeting too small dataset
        if len(dataset) < self.num_replicas * samples_per_gpu:
            raise ValueError(
                'You may use too small dataset and our distributed '
                'sampler cannot pad your dataset correctly. We highly '
                'recommend you to use fewer GPUs to finish your work')
        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed)

    def update_sampler(self, dataset, samples_per_gpu=None):
        self.dataset = dataset
        if samples_per_gpu is not None:
            self.samples_per_gpu = samples_per_gpu
        # fix the bug of the official implementation
        self.num_samples_per_replica = int(
            int(
                np.ceil(
                    len(self.dataset) * 1.0 / self.num_replicas /
                    self.samples_per_gpu)))
        self.num_samples = self.num_samples_per_replica * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

        # to avoid padding bug when meeting too small dataset
        if len(dataset) < self.num_replicas * self.samples_per_gpu:
            raise ValueError(
                'You may use too small dataset and our distributed '
                'sampler cannot pad your dataset correctly. We highly '
                'recommend you to use fewer GPUs to finish your work')

    def __iter__(self):
        # default training setting of MMSegmentation is iter-based
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            # When :attr:`shuffle=True`, this ensures all replicas
            # use a different random ordering for each epoch.
            # Otherwise, the next iteration of this sampler will
            # yield the same ordering.
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
