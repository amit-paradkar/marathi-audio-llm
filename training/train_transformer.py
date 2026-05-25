import glob
import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from models.tiny_transformer import model


class TokenDataset(Dataset):

    def __init__(self, files):

        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        tokens = np.load(self.files[idx])

        tokens = torch.tensor(tokens).long()

        x = tokens[:-1]
        y = tokens[1:]

        return x, y


#old definition def train_transformer():
def train_transformer(
    epochs=5,
    batch_size=2,
    lr=1e-4
):

    files = glob.glob("data/tokenized/*.npy")

    dataset = TokenDataset(files)

    loader = DataLoader(dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    for epoch in range(epochs):

        for x, y in loader:

            outputs = model(x)

            logits = outputs.logits

            loss = F.cross_entropy(
                logits.view(-1, 1024),
                y.view(-1)
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            print(loss.item())

    torch.save(
        model.state_dict(),
        "tiny_transformer.pt"
    )