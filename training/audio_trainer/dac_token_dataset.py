import glob
import torch
import random
from torch.utils.data import Dataset

SEQUENCE_LENGTH = 1024

class DACTokenDataset(Dataset):

    def __init__(self, token_dir, seq_len=1024, limit=None):

        self.files = sorted(
            glob.glob(f"{token_dir}/*.pt")
        )
        self.seq_len = seq_len
        if limit:
            self.files = self.files[:limit]

    def __len__(self):
        return len(self.files)

    
    def __getitem__(
        self,
        idx
    ):

        tokens = torch.load(
            self.files[idx]
        ).flatten()

        if len(tokens) < self.seq_len + 1:

            raise ValueError(
                f"{self.files[idx]} "
                f"too short"
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

        x = chunk[:-1]

        y = chunk[1:]

        return x, y