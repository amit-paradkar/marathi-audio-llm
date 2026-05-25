import librosa
import numpy as np
from config.audio_config import CHUNK_SIZE




def load_audio(path):

    audio, sr = librosa.load(
        path,
        sr=16000,
        mono=True
    )

    return audio


def chunk_audio(
    audio,
    chunk_size=CHUNK_SIZE
):
#def chunk_audio(audio, chunk_size=16000):

    chunks = []

    for i in range(0, len(audio), chunk_size):

        chunk = audio[i:i + chunk_size]

        if len(chunk) == chunk_size:
            chunks.append(chunk)

    return np.array(chunks)