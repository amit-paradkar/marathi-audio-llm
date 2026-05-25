import torch
import numpy as np
import glob
import os

from models.dac import TinyDAC
from scripts.preprocess_audio import load_audio


model = TinyDAC()
model.load_state_dict(torch.load("tiny_dac.pt"))
model.eval()


RAW_DIR = "data/raw"
TOKEN_DIR = "data/tokenized"

os.makedirs(TOKEN_DIR, exist_ok=True)


files = glob.glob(f"{RAW_DIR}/*.wav")


for idx, filepath in enumerate(files):

    audio = load_audio(filepath)

    audio = torch.tensor(audio).float()

    audio = audio.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        tokens = model.encode(audio)

    tokens = tokens.squeeze().numpy()

    np.save(
        f"{TOKEN_DIR}/sample_{idx}.npy",
        tokens
    )