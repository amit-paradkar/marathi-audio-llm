# before execting install the following
# pip install librosa soundfile pandas tqdm
# librosa library required installation of llvm@20
# a. brew install llvm@20  
# b. export LLVM_CONFIG=$(brew --prefix llvm)/bin/llvm-config   
#.   export LLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm
#.   export CMAKE_PREFIX_PATH=$(brew --prefix llvm)
#Check llvm version $LLVM_CONFIG --version 
# brew link llvm@20 --force
# uv pip install --no-cache-dir llvmlite --index-url https://pypi.org/simple
# uv pip install --no-cache librosa
# uv pip install --no-cache soundfile pandas tqdm --index-url https://pypi.org/simple
#and execute using 
# python scripts/preprocess_dataset.py
# This converts the mp3 into wav format.
# Next perform chunking
import os
import glob
import librosa
import soundfile as sf

from tqdm import tqdm


RAW_DIR = "data/raw/cv-corpus-25.0-2026-03-09/mr/clips"
OUT_DIR = "data/processed"

os.makedirs(OUT_DIR, exist_ok=True)


files = glob.glob(f"{RAW_DIR}/*.mp3")


for filepath in tqdm(files):

    filename = os.path.basename(filepath)

    audio, sr = librosa.load(
        filepath,
        sr=16000,
        mono=True
    )

    outpath = (
        f"{OUT_DIR}/"
        + filename.replace(".mp3", ".wav")
    )

    sf.write(
        outpath,
        audio,
        16000
    )