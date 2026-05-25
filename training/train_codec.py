import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.dac import TinyDAC


def train_codec(dataset):

    loader = DataLoader(dataset, batch_size=4)

    model = TinyDAC()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()

    for epoch in range(5):

        for audio in loader:

            reconstructed, tokens = model(audio)

            loss = F.mse_loss(reconstructed, audio)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            print(f"loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "tiny_dac.pt")