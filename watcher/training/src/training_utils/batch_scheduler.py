"""A batch size scheduler class"""

import math
import gc
from typing import Iterator
from torch import Tensor
from torch.utils.data import DataLoader


class BatchScheduler(object):
    """A class that linearly increases the batch size during the training.

    Because changing batch sizes during the training makes estimating the number of total steps complex,
    this class computes the estimated total steps, and keeps it as one of its attributes. Refer to the value
    if you need the value, for example, for LR schedulers.
    _step() is implicitly called during iteration, so do not call it explicitly.
    Example:
        dataloader = DataLoader(your_data, ...)
        batch_scheduler = BatchSizeScheduler(dataloader, args,...)
        total_steps = batch_scheduler.estimated_total_steps
        scheduler = CustomScheduler(total_steps)
        ...
        for batch in batch_scheduler:
            input, label = batch,
            pred = model(input)
            loss = criteria(pred, loss)
            loss.backward()
            optimizer.step()
            scheduler.step()

    """

    def __init__(
        self,
        dataloader: DataLoader,
        total_epochs: int,
        world_size: int,
        step_scale: int = 8,
        active_phase: int = 0.02,
        min_batch_size: int = None,
        enabled=True,
    ) -> None:
        if min_batch_size is None:
            min_batch_size = step_scale
        if not isinstance(dataloader, DataLoader):
            raise TypeError("dataloader must be a DataLoader object.")
        max_batch_size = dataloader.batch_size
        self.enabled = enabled
        self.dataloader = dataloader
        self.step_scale = step_scale
        self.step_num = 0
        self.slice_fn = None
        n_data_per_epoch = len(self.dataloader.dataset) // world_size
        self.total_data_points = int(n_data_per_epoch * total_epochs)
        if enabled:
            if min_batch_size >= max_batch_size:
                raise ValueError(
                    "min_batch_size must be smaller than the dataloader's batch size."
                )
            if step_scale >= max_batch_size:
                raise ValueError(
                    "step_scale must be smaller than the dataloader's batch size."
                )
            if step_scale <= 0:
                raise ValueError("step_scale must be a positive, non-zero integer")
            if max_batch_size % step_scale != 0:
                raise ValueError(
                    "The dataloader's batch_size must be divisible by step_scale."
                )
            if max_batch_size % min_batch_size != 0:
                raise ValueError(
                    "The dataloader's batch_size must be divisible by min_batch_size."
                )
            if min_batch_size % step_scale != 0:
                raise ValueError("min_batch_size must be divisible by step_scale.")
            if (max_batch_size - min_batch_size) % step_scale != 0:
                raise ValueError(
                    "Difference between the dataloader's batch_size and min_batch_size must be divisible by step_scale."
                )
            # Create a series of batch sizes
            self.batch_size_sequence = self._get_batch_size_sequence(
                max_batch_size, min_batch_size, step_scale
            )
            # Estimate the number of total steps and steps for each batch size step
            # NOTE: This estimation of total batch size should be slightly different from the actual total steps.
            #       There are number of factors that makes it difficult to calculate the steps simply.
            #       However, because the difference is considered small enough, maintainability and simplicity is prioritized
            #       over the precision of estimated total steps.

            active_data_points = int(active_phase * self.total_data_points)
            inactive_data_points = self.total_data_points - active_data_points
            batch_size_sum = sum(self.batch_size_sequence[:-1])
            self.steps_per_batch_size = math.ceil(active_data_points / batch_size_sum)
            self.estimated_total_steps = (
                self.steps_per_batch_size * len(self.batch_size_sequence[:-1])
                + inactive_data_points // max_batch_size
            )
            # Set the starting batch size
            self.batch_size = self.batch_size_sequence[0]
            self.progress = 0
            if len(self.batch_size_sequence) == 1:
                self.enabled = False
        else:
            self.batch_size = max_batch_size
            self.estimated_total_steps = self.total_data_points // max_batch_size
            self.batch_size_sequence = [max_batch_size]
            self.steps_per_batch_size = self.estimated_total_steps
            self.progress = 0

    def _get_batch_size_sequence(
        self, max_batch_size: int, min_batch_size: int, step_scale: int
    ) -> list[int]:
        """Yields batch sizes that are devisors of max_batch_size."""
        divisors = []
        for i in range(1, int(max_batch_size / step_scale) + 1):
            candidate = step_scale * i
            if candidate >= min_batch_size:
                if max_batch_size % candidate == 0:
                    divisors.append(int(candidate))
        return divisors

    def _slice_single_tensor(self, batch: Tensor) -> Iterator:
        for subbatch in batch.split(self.batch_size):
            yield subbatch

    def _slice_tensor_tuple(self, batch: tuple[Tensor, ...]) -> Iterator:
        sliced_elements = [t.split(self.batch_size) for t in batch]
        for subbatch in zip(*sliced_elements):
            yield subbatch

    def _slice_tensor_list(self, batch: list[Tensor, ...]) -> Iterator:
        sliced_elements = [t.split(self.batch_size) for t in batch]
        for subbatch in zip(*sliced_elements):
            yield list(subbatch)

    def _step(self):
        """Counts up one step."""
        if self.enabled:
            self.step_num += 1
            if self.step_num % self.steps_per_batch_size == 0:
                self.progress += 1
                if self.progress < len(self.batch_size_sequence):
                    self.batch_size = self.batch_size_sequence[self.progress]
                    if self.batch_size == len(self.batch_size_sequence) - 1:
                        self.enabled = False
                    gc.collect()
                else:
                    self.enabled = False
        else:
            pass

    def __iter__(self) -> Iterator:
        if self.enabled:
            # Check the type of a batch by looking at the first one
            if self.slice_fn is None:
                for sample_batch in self.dataloader:
                    if isinstance(sample_batch, tuple) | isinstance(sample_batch, list):
                        for e in sample_batch:
                            if not isinstance(e, Tensor):
                                raise ValueError(
                                    "All elements of batches yielded by the dataloader must be torch.Tensor."
                                )
                        if isinstance(sample_batch, tuple):
                            self.slice_fn = self._slice_tensor_tuple
                        else:
                            self.slice_fn = self._slice_tensor_list
                    elif isinstance(sample_batch, Tensor):
                        self.slice_fn = self._slice_single_tensor
                    else:
                        raise ValueError(
                            "A batch yielded by the dataloader must be a tensor, or a tuple/tuple of tensors."
                        )
                    # Yield the first batch
                    for subbatch in self.slice_fn(sample_batch):
                        yield subbatch
                        self._step()
                    break

            # The main loop
            for batch in self.dataloader:
                for subbatch in self.slice_fn(batch):
                    yield subbatch
                    self._step()

            # Place a 'return' here to exit the loop
            # NOTE: If there is not 'return' here, the batch-scheduler reload the dataloader again and again.
            return

        else:
            yield from self.dataloader

    def estimate_n_steps(self, fraction: float) -> int:
        """Estimate the number of steps to achive the given fraction of data processed.
        Args:
            fraction (float): Target fraction of data.
                For example, pass 0.01 if you want to know the number of steps to achive 1% of data used.
        Returns:
            n_steps (int): Estimated number of steps.
        """
        n_achieved = 0
        n_steps = 0
        target = math.floor(fraction * self.total_data_points)
        for bs in self.batch_size_sequence[:-1]:
            n_in_this_phase = self.steps_per_batch_size * bs
            if n_achieved + n_in_this_phase >= target:
                diff = target - n_achieved
                n_steps += diff // bs
                break
            n_achieved += n_in_this_phase
            n_steps += self.steps_per_batch_size
        else:
            bs = self.batch_size_sequence[-1]
            diff = target - n_achieved
            n_steps += diff // bs
        return n_steps

    def state_dict(self) -> dict:
        """Saves the current state of the batch scheduler in a dictionary"""
        state = {
            "enabled": self.enabled,
            "step_scale": self.step_scale,
            "step_num": self.step_num,
            "total_data_points": self.total_data_points,
            "batch_size": self.batch_size,
            "progress": self.progress,
            "batch_size_sequence": self.batch_size_sequence,
            "steps_per_batch_size": self.steps_per_batch_size,
            "estimated_total_steps": self.estimated_total_steps,
        }
        return state

    def load_state_dict(self, state_dict: dict):
        """Loads the past state of the batch scheudler.
        This method updates attributes in-place.
        Args:
            state_dict (dict): Dictionary created by self.state_dict().
        """
        self.enabled = state_dict["enabled"]
        self.step_scale = state_dict["step_scale"]
        self.step_num = state_dict["step_num"]
        self.total_data_points = state_dict["total_data_points"]
        self.batch_size = state_dict["batch_size"]
        self.progress = state_dict["progress"]
        self.batch_size_sequence = state_dict["batch_size_sequence"]
        self.steps_per_batch_size = state_dict["steps_per_batch_size"]
        self.estimated_total_steps = state_dict["estimated_total_steps"]
