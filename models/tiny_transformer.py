from transformers import GPT2Config
from transformers import GPT2LMHeadModel


config = GPT2Config(
    vocab_size=1024,
    n_positions=256,
    n_embd=256,
    n_layer=4,
    n_head=4
)


model = GPT2LMHeadModel(config)