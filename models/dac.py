import torch
import torch.nn as nn


class TinyDAC(nn.Module):

    def __init__(self, vocab_size=1024):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 4, stride=2, padding=1),
        )

        self.codebook = nn.Embedding(vocab_size, 256)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, 4, stride=2, padding=1),
        )

    def encode(self, audio):

        z = self.encoder(audio)

        # fake quantization for understanding
        tokens = torch.argmax(z, dim=1)

        return tokens

    def decode(self, tokens):

        z_q = self.codebook(tokens)

        z_q = z_q.permute(0, 2, 1)

        audio = self.decoder(z_q)

        return audio

    def forward(self, audio):

        tokens = self.encode(audio)

        reconstructed = self.decode(tokens)

        return reconstructed, tokens