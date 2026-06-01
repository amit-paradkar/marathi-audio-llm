import torch
import torch.nn as nn

class AudioTransformer(nn.Module):

    def __init__(
        self,
        vocab_size=1024,
        d_model=256,
        n_heads=4,
        n_layers=4,
        max_len=1024
    ):
        super().__init__()

        self.token_emb = nn.Embedding(
            vocab_size,
            d_model
        )

        self.pos_emb = nn.Embedding(
            max_len,
            d_model
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=n_layers
        )

        self.lm_head = nn.Linear(
            d_model,
            vocab_size
        )

        # Create once, reuse forever
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(
                    max_len,
                    max_len
                ),
                diagonal=1
            ).bool()
        )

    def forward(self, x):

        B, T = x.shape

        pos = torch.arange(
            T,
            device=x.device
        ).unsqueeze(0)

        h = (
            self.token_emb(x)
            + self.pos_emb(pos)
        )

        mask = self.causal_mask[
            :T,
            :T
        ]

        h = self.transformer(
            h,
            mask=mask
        )

        logits = self.lm_head(h)

        return logits

    '''def forward(self, x):

        B, T = x.shape

        pos = torch.arange(
            T,
            device=x.device
        ).unsqueeze(0)

        h = (
            self.token_emb(x)
            + self.pos_emb(pos)
        )

        mask = torch.triu(
            torch.ones(T, T, device=x.device),
            diagonal=1
        ).bool()

        h = self.transformer(
            h,
            mask=mask
        )

        return self.lm_head(h)'''

    '''def forward(self, x):

        B, T = x.shape

        pos = torch.arange(
            T,
            device=x.device
        ).unsqueeze(0)

        h = (
            self.token_emb(x)
            + self.pos_emb(pos)
        )

        h = self.transformer(h)

        return self.lm_head(h)'''