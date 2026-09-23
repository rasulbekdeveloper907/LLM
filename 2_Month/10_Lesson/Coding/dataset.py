from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BinaryTokenDataset(Dataset):
    """Random fixed-length samples from a binary token stream."""

    def __init__(self, file_path, block_size):
        self.file_path = Path(file_path)
        self.block_size = block_size

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found:\n{self.file_path}"
            )

        self.data = np.memmap(
            self.file_path,
            dtype=np.uint16,
            mode="r",
        )

        if len(self.data) <= block_size:
            raise ValueError(
                f"Dataset is too small for block_size={block_size}"
            )

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, index):
        x = self.data[
            index:index + self.block_size
        ]

        y = self.data[
            index + 1:index + self.block_size + 1
        ]

        x = torch.from_numpy(
            np.asarray(x, dtype=np.int64)
        )

        y = torch.from_numpy(
            np.asarray(y, dtype=np.int64)
        )

        return x, y