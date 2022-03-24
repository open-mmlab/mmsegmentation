# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import math
from typing import Iterator, Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Sampler

from mmseg.core.utils import sync_random_seed


class DistributedSampler(_DistributedSampler):
    """DistributedSampler inheriting from
    `torch.utils.data.DistributedSampler`.

    In pytorch of lower versions, there is no `shuffle` argument. This child
    class will port one to DistributedSampler.

    Args:
        datasets (Dataset): the dataset will be loaded.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, world_size is retrieved from the
            current distributed group.
        rank (int, optional):  Rank of the current process within num_replicas.
            By default, rank is retrieved from the current distributed group.
        shuffle (bool): If True (default), sampler will shuffle the indices.
        seed (int): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
    """

    def __init__(self,
                 dataset: Dataset,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed=0) -> None:
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed)

    def __iter__(self) -> Iterator:
        """
         Yields:
            Iterator: iterator of indices for rank.
        """
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            # When :attr:`shuffle=True`, this ensures all replicas
            # use a different random ordering for each epoch.
            # Otherwise, the next iteration of this sampler will
            # yield the same ordering.
            g.manual_seed(self.epoch + self.seed)
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


class MixedBatchDistributedSampler(Sampler):
    """Distributed Sampler for mixed data batch.

    Args:
        datasets (list): List of datasets will be loaded.
        sample_ratio (list): List of the ratio of each dataset in a batch, e.g.
            datasets=[DatasetA, DatasetB], sample_ratio=[0.25, 0.75],
            sample_per_gpu=1, gpus=8, it means 2 gpus load DatasetA, and 6 gpus
            load DatasetB. The length of datasets must be equal to length of
            sample_ratio.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, world_size is retrieved from the
            current distributed group.
        rank (int, optional):  Rank of the current process within num_replicas.
            By default, rank is retrieved from the current distributed group.
        shuffle (bool): If True (default), sampler will shuffle the indices.
        seed (int): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
    """

    def __init__(self,
                 datasets: Sequence[Dataset],
                 sample_ratio: Sequence[float],
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0) -> None:

        # base class `Sampler` do nothing in `__init__` function
        # super().__init__()

        assert len(datasets) == len(sample_ratio)
        assert sum(sample_ratio) == 1.

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            rank = dist.get_rank()

        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f'Invalid rank {rank}, rank should be in the interval'
                f' [0, {num_replicas - 1}]')

        self.datasets = datasets
        self.num_replicas = num_replicas

        self.datasets_num_replicas = [
            math.ceil(num_replicas * r) for r in sample_ratio
        ]
        self.cumulative_replicas = []
        t = 0
        for nr in self.datasets_num_replicas:
            t += nr
            self.cumulative_replicas.append(t)
        self.datasets_length = [len(d) for d in datasets]
        self.cumulative_datasets_length = []

        t = 0
        for dl in self.datasets_length:
            t += dl
            self.cumulative_datasets_length.append(t)

        # the smallest num_sample
        self.datasets_num_samples = [
            math.ceil(length / ratio) for length, ratio in zip(
                self.datasets_length, self.datasets_num_replicas)
        ]
        self.num_samples = min(self.datasets_num_samples)

        # the dataset that decides the num_samples and total_size
        self.key_dataset = self.datasets_num_samples.index(self.num_samples)
        self.key_dataset_length = self.datasets_length[self.key_dataset]
        self.total_size = [
            self.num_samples * nr for nr in self.datasets_num_replicas
        ]

        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed)

    def __iter__(self) -> Iterator:
        """
        Yields:
            Iterator: iterator of indices for current rank.
        """

        # datasets map different rank
        for dataset_idx, cumulative_replicas_ in enumerate(
                self.cumulative_replicas):
            if self.rank < cumulative_replicas_:
                break

        # deterministically shuffle each datasets based on epoch
        if self.shuffle:
            g = torch.Generator()
            # When :attr:`shuffle=True`, this ensures all replicas
            # use a different random ordering for each epoch.
            # Otherwise, the next iteration of this sampler will
            # yield the same ordering.
            g.manual_seed(self.epoch + self.seed)

            indices = torch.randperm(
                self.datasets_length[dataset_idx], generator=g).tolist()
        else:
            indices = torch.arange(self.datasets_length[dataset_idx]).tolist()

        if self.total_size[dataset_idx] > len(indices):
            # add extra samples for key_dataset to make it evenly divisible
            indices += indices[:(self.total_size[dataset_idx] - len(indices))]
        else:
            indices = indices[:self.total_size[dataset_idx]]

        assert len(indices) == self.total_size[dataset_idx]

        # subsample
        last_cumulative_replicas = 0 \
            if dataset_idx == 0 else self.cumulative_replicas[dataset_idx - 1]

        indices = indices[(
            self.rank - last_cumulative_replicas
        ):self.total_size[dataset_idx]:self.datasets_num_replicas[dataset_idx]]

        assert len(indices) == self.num_samples

        # find the dataset for this rank
        last_cumulative_length = 0 \
            if dataset_idx == 0 else \
            self.cumulative_datasets_length[dataset_idx-1]

        indices = [idx + last_cumulative_length for idx in indices]

        return iter(indices)

    def __len__(self) -> int:
        """Get the combined dataset length."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this
        ensures all replicas use a different random ordering for each epoch.
        Otherwise, the next iteration of this sampler will yield the same
        ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
