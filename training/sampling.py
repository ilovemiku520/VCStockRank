"""Keep full daily cross-sections together while packing several days per batch."""
from collections import defaultdict
import math
import torch

class DateBatchSampler:
    def __init__(self, dataset, batch_size, shuffle=False):
        groups = defaultdict(list)
        for index, sample in enumerate(dataset.samples):
            groups[sample['date']].append(index)
        self.groups = [groups[date] for date in sorted(groups)]
        self.days_per_batch = max(1, batch_size // max(map(len, self.groups), default=1))
        self.shuffle = shuffle

    def __iter__(self):
        order = torch.randperm(len(self.groups)).tolist() if self.shuffle else range(len(self.groups))
        batch = []
        days = 0
        for index in order:
            batch.extend(self.groups[index])
            days += 1
            if days == self.days_per_batch:
                yield batch
                batch, days = [], 0
        if batch:
            yield batch

    def __len__(self):
        return math.ceil(len(self.groups) / self.days_per_batch)
