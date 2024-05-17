import torch
import torch.distributed

import src.utils.distributed as dist_utils


class AverageMeter:
    """Computes and stores the average and current value of a given metric. Supports
    distributed training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.synced_count = 0
        self.new_count = 0

        self.synced_sum = 0
        self.new_sum = 0

    def update(self, val, n=1):
        self.new_sum += val * n
        self.new_count += n

    def _sync(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            new_counts = sum(dist_utils.do_all_gather_object(self.new_count))
            new_sums = sum(dist_utils.do_all_gather_object(self.new_sum))
        else:
            new_counts, new_sums = self.new_count, self.new_sum

        self.synced_count += new_counts
        self.synced_sum += new_sums

        self.new_count = 0
        self.new_sum = 0

    @property
    def avg(self):
        self._sync()
        # No need to add new_count since call to avg() forces a call to _sync()
        return self.synced_sum / self.synced_count


class BestMeter:
    """Computes and stores the best observed value of a metric."""

    def __init__(self, direction="max"):
        assert direction in {"max", "min"}
        self.direction = direction
        self.reset()

    def reset(self):
        if self.direction == "max":
            self.val = -float("inf")
        else:
            self.val = float("inf")

    def update(self, val):
        """Update meter and return boolean flag indicating if the current value is
        the best so far."""

        if self.direction == "max":
            if val > self.val:
                self.val = val
                return True
        elif self.direction == "min":
            if val < self.val:
                self.val = val
                return True

        return False
