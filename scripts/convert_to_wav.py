import librosa
import soundfile as sf
import os

input_dir = "/content/raw_audio"
output_dir = "/content/data/marathi"

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):

    if file.endswith(".mp3") or file.endswith(".wav"):

        path = os.path.join(input_dir, file)

        audio, sr = librosa.load(path, sr=16000, mono=True)

        out_path = os.path.join(output_dir, file.replace(".mp3", ".wav"))

        sf.write(out_path, audio, 16000)