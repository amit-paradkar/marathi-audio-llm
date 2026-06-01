'''
Reads data/chunks_by_silence and creates tokens in tokens_by_silence.

'''
import os
import glob
import torch
import torchaudio

from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

# DAC imports
from dac.model import DAC
from dac.utils import download


# =========================================================
# CONFIG
# =========================================================

INPUT_DIR = "data/chunks_by_silence"
OUTPUT_DIR = "data/tokens_by_silence"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4

TARGET_SR = 24000


# =========================================================
# LOAD DAC MODEL
# =========================================================

print("Loading DAC model...")

model_path = download(model_type="24khz")

model = DAC.load(model_path)

model.to(DEVICE)

model.eval()

print("DAC ready")


# =========================================================
# LOAD FILES
# =========================================================

files = glob.glob(f"{INPUT_DIR}/*.wav")

print(f"Found {len(files)} wav files")


# =========================================================
# HELPER FUNCTION
# =========================================================

def load_audio(filepath):

    audio, sr = torchaudio.load(filepath)

    # mono
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # resample
    if sr != TARGET_SR:

        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=TARGET_SR
        )

        audio = resampler(audio)

    # remove channel dimension
    audio = audio.squeeze(0)

    return audio


# =========================================================
# BATCH LOOP
# =========================================================

for batch_start in tqdm(range(0, len(files), BATCH_SIZE)):

    batch_files = files[
        batch_start: batch_start + BATCH_SIZE
    ]

    audios = []

    valid_files = []

    # ---------------------------------------------
    # LOAD AUDIO
    # ---------------------------------------------

    for filepath in batch_files:

        try:

            audio = load_audio(filepath)

            audios.append(audio)

            valid_files.append(filepath)

        except Exception as e:

            print(f"ERROR loading {filepath}")
            print(e)

    if len(audios) == 0:
        continue

    # ---------------------------------------------
    # PAD TO SAME LENGTH
    # ---------------------------------------------

    padded = pad_sequence(
        audios,
        batch_first=True
    )

    # add channel dimension
    padded = padded.unsqueeze(1)

    padded = padded.to(DEVICE)

    # ---------------------------------------------
    # DAC ENCODE
    # ---------------------------------------------

    try:

        with torch.no_grad():

            z, codes, latents, _, _ = model.encode(padded)

    except Exception as e:

        print("Batch encode failed")
        print(e)

        continue

    # ---------------------------------------------
    # SAVE TOKENS
    # ---------------------------------------------

    for i, filepath in enumerate(valid_files):

        try:

            base = os.path.splitext(
                os.path.basename(filepath)
            )[0]

            outpath = os.path.join(
                OUTPUT_DIR,
                f"{base}.pt"
            )

            # save one sample's tokens
            sample_codes = codes[i].cpu()

            torch.save(
                sample_codes,
                outpath
            )

        except Exception as e:

            print(f"ERROR saving {filepath}")
            print(e)

print("Batch tokenization complete")