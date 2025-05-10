from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset, Sampler


def read_data(filename: str, num_feature: int) -> tuple[Tensor, Tensor]:
    with open(filename) as file:
        lines = file.readlines()

    x = []
    y = []
    for line in lines:
        parts = line.strip().split()
        y.append(float(parts[0]))
        features = torch.zeros(num_feature, dtype=torch.float)
        for feature in parts[1:]:
            index, value = feature.split(':')
            features[int(index) - 1] = float(value)
        x.append(features)

    return torch.stack(x), torch.tensor(y, dtype=torch.float)


class SVMDataset(Dataset):
    def __init__(self, x: Tensor, y: Tensor, device: torch.device | str):
        self.x = x.to(device)
        self.y = y.to(device)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int | Tensor) -> tuple[Tensor, Tensor]:
        return self.x[idx], self.y[idx]


class BatchedTensorSampler(Sampler[torch.Tensor]):
    def __init__(
            self,
            data_source,
            batch_size: int,
    ) -> None:
        """
        Args:
            data_source: Dataset to sample from
            batch_size: Size of each batch
        """
        super().__init__()
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[torch.Tensor]:
        n = len(self.data_source)
        indices = torch.randperm(n)

        for i in range(0, n, self.batch_size):
            end = i + self.batch_size
            if end >= n:
                end = n - 1
            batch_indices = indices[i:end]
            yield batch_indices

    def __len__(self) -> int:
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size


class TensorDataloader(IterableDataset):
    def __init__(self, dataset: SVMDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        sampler = BatchedTensorSampler(self.dataset, self.batch_size)
        for indices in sampler:
            yield self.dataset[indices]
