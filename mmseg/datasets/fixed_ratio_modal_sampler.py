"""
Fixed Ratio Modal Sampler for Multi-Dataset Training - MMSeg 1.x Version

Registered as DATA_SAMPLERS for mmengine compatibility.
"""
from typing import Dict, Iterator, List, Optional, Sequence, Sized, Union

import torch
from torch.utils.data import Sampler

from mmseg.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class FixedRatioModalSampler(Sampler):
    """Fixed ratio modal sampler.

    Ensures each batch has a fixed modal composition.

    Args:
        dataset: Dataset object, must have data_list with modal_type.
        modal_ratios: Modal sampling ratios.
        modal_order: Modal ordering.
        reference_modal: Reference modal for epoch length calculation.
        seed: Random seed.
        batch_size: Batch size (replaces samples_per_gpu).
    """

    def __init__(
        self,
        dataset: Sized,
        modal_ratios: Optional[Union[Sequence[int], Dict[str, int]]] = None,
        modal_order: Optional[Sequence[str]] = None,
        reference_modal: Optional[str] = None,
        seed: Optional[int] = None,
        batch_size: int = 1,
        **kwargs,
    ):
        if modal_ratios is None:
            raise ValueError(
                "modal_ratios must be provided for FixedRatioModalSampler.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = 0 if seed is None else seed
        self.epoch = 0
        self.reference_modal = reference_modal

        if isinstance(modal_ratios, dict):
            if modal_order is None:
                modal_order = list(modal_ratios.keys())
            self.modal_order = list(modal_order)
            self.modal_ratios = [
                modal_ratios[modal] for modal in self.modal_order]
        else:
            if modal_order is None:
                raise ValueError(
                    "modal_order must be provided when "
                    "modal_ratios is a list.")
            self.modal_order = list(modal_order)
            self.modal_ratios = list(modal_ratios)

        self._validate_ratios()
        self.modal_indices = self._group_by_modal()
        self.num_samples = self._calculate_num_samples()

        self._print_statistics()

    def _validate_ratios(self):
        if len(self.modal_order) != len(self.modal_ratios):
            raise ValueError(
                "modal_order and modal_ratios must have the same length.")
        if any(ratio <= 0 for ratio in self.modal_ratios):
            raise ValueError("All modal ratios must be positive.")
        if sum(self.modal_ratios) > self.batch_size:
            raise ValueError(
                "Sum of modal ratios must be <= batch_size "
                f"(got {sum(self.modal_ratios)} > {self.batch_size}).")
        if self.batch_size % sum(self.modal_ratios) != 0:
            raise ValueError(
                "batch_size must be divisible by sum(modal_ratios) "
                f"(got {self.batch_size} % {sum(self.modal_ratios)} != 0).")

    def _group_by_modal(self) -> Dict[str, List[int]]:
        modal_indices = {modal: [] for modal in self.modal_order}
        # In 1.x, dataset.data_list contains the data info dicts
        for idx in range(len(self.dataset)):
            data_info = self.dataset.get_data_info(idx)
            modal_type = data_info.get('modal_type', 'unknown')
            if modal_type in modal_indices:
                modal_indices[modal_type].append(idx)
        return modal_indices

    def _calculate_num_samples(self) -> int:
        total_ratio = sum(self.modal_ratios)
        batch_repeats = self.batch_size // total_ratio

        if self.reference_modal is None:
            total_original = sum(
                len(v) for v in self.modal_indices.values())
            full_batches = total_original // self.batch_size
            return full_batches * self.batch_size

        if self.reference_modal not in self.modal_order:
            raise ValueError(
                f"reference_modal '{self.reference_modal}' "
                f"is not in modal_order.")

        reference_count = len(self.modal_indices[self.reference_modal])
        if reference_count == 0:
            raise ValueError(
                f"No samples found for reference_modal "
                f"'{self.reference_modal}'.")

        reference_ratio = self.modal_ratios[
            self.modal_order.index(self.reference_modal)]
        total_groups = reference_count // reference_ratio
        full_batches = total_groups // batch_repeats
        return full_batches * self.batch_size

    def _print_statistics(self):
        print("\n" + "=" * 60)
        print("Fixed Ratio Modal Sampler Statistics")
        print("=" * 60)
        print(f"Batch size: {self.batch_size}")
        print("Modal Ratios:")
        for modal, ratio in zip(self.modal_order, self.modal_ratios):
            print(f"  {modal}: {ratio}")
        print("Modal Distribution (Original):")
        total_original = sum(len(v) for v in self.modal_indices.values())
        for modal in self.modal_order:
            count = len(self.modal_indices[modal])
            percentage = (
                (count / total_original * 100)
                if total_original > 0 else 0.0)
            print(f"  {modal}: {count:5d} samples ({percentage:5.2f}%)")
        if self.reference_modal is not None:
            reference_count = len(
                self.modal_indices[self.reference_modal])
            reference_ratio = self.modal_ratios[
                self.modal_order.index(self.reference_modal)]
            print(
                f"Reference modal: {self.reference_modal} "
                f"(count={reference_count}, ratio={reference_ratio})")
        print(f"Total samples per epoch: {self.num_samples}")
        print(f"Iterations per epoch: {self.num_samples // self.batch_size}")
        print("=" * 60 + "\n")

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        per_modal_pools = {}
        per_modal_pos = {}
        for modal, indices in self.modal_indices.items():
            if len(indices) == 0:
                raise ValueError(
                    f"No samples found for modal '{modal}'.")
            perm = torch.randperm(len(indices), generator=g)
            per_modal_pools[modal] = [indices[i] for i in perm.tolist()]
            per_modal_pos[modal] = 0

        indices = []
        batch_repeats = self.batch_size // sum(self.modal_ratios)
        num_batches = self.num_samples // self.batch_size

        for _ in range(num_batches):
            batch_indices = []
            for _ in range(batch_repeats):
                for modal, ratio in zip(
                        self.modal_order, self.modal_ratios):
                    pool = per_modal_pools[modal]
                    pos = per_modal_pos[modal]
                    for _ in range(ratio):
                        if pos >= len(pool):
                            perm = torch.randperm(
                                len(pool), generator=g)
                            pool = [pool[i] for i in perm.tolist()]
                            per_modal_pools[modal] = pool
                            pos = 0
                        batch_indices.append(pool[pos])
                        pos += 1
                    per_modal_pos[modal] = pos

            indices.extend(batch_indices)

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
