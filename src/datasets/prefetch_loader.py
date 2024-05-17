from functools import partial

import torch


class PrefetchLoader:
    """A wrapper around a Pytorch dataloader that prefetches the next batch on the GPU."""

    def __init__(self, dataloader, device, normalize_in_prefetcher=False, mean=None, std=None):
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.stream_context = partial(torch.cuda.stream, stream=self.stream)
        self.normalize_in_prefetcher = normalize_in_prefetcher
        if self.normalize_in_prefetcher:
            assert mean is not None and std is not None
            self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1).mul_(255)
            self.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1).mul_(255)

    def __iter__(self):
        first = True
        current_input = None
        current_target = None

        for next_input, next_target in self.dataloader:
            with self.stream_context():
                next_input = next_input.to(device=self.device, dtype=torch.float32, non_blocking=True)
                if self.normalize_in_prefetcher:
                    next_input.sub_(self.mean).div_(self.std)

                next_target = next_target.to(device=self.device, non_blocking=True)

            if not first:
                yield current_input, current_target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(self.stream)

            current_input = next_input
            current_target = next_target

        yield current_input, current_target

    def __len__(self):
        return len(self.dataloader)

    @property
    def sampler(self):
        return self.dataloader.sampler

    @property
    def dataset(self):
        return self.dataloader.dataset
