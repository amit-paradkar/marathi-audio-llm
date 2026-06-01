# shard_dataset.py

import random
import torch
from torch.utils.data import Dataset


class ShardDataset(Dataset):

    def __init__(
        self,
        shard,
        seq_len=1024
    ):

        self.shard = shard
        self.seq_len = seq_len

    def __len__(self):

        return len(self.shard)

    def __getitem__(
        self,
        idx
    ):

        tokens = (
            self.shard[idx]
            .reshape(-1)
            .long()
        )

        start = random.randint(
            0,
            len(tokens)
            - self.seq_len
            - 1
        )

        chunk = tokens[
            start:
            start + self.seq_len + 1
        ]

        return (
            chunk[:-1],
            chunk[1:]
        )