#Chunking method uses ff
'''
# Setup ffmpeg
brew install ffmpeg
 
Difference between pydub and ffmpeg chunking

| Feature                   | Pydub       | FFmpeg            |
| ------------------------- | ----------- | ----------------- |
| Speed                     | Slower      | Much faster       |
| Memory usage              | High        | Very low          |
| Large datasets            | Less ideal  | Excellent         |
| Uses Python RAM           | Yes         | Minimal           |
| Native optimization       | No          | Yes               |
| Streaming capable         | Limited     | Excellent         |
| GPU dependency            | None        | None              |
| Apple Silicon performance | Good        | Excellent         |
| Fine audio manipulation   | Easier      | Harder            |
| Production pipelines      | Less common | Industry standard |

1. Memory Difference
    Pydub
    audio = AudioSegment.from_wav(filepath)
    Loads entire file.
    Example:
    - 1 hour WAV hundreds of MB in RAM
    Problematic for:
    - huge datasets, many parallel files

    FFmpeg
    - Processes incrementally/streamingly.
    Can split:
    - hours of audio
    - thousands of files
    - without huge memory spikes.

    | Task           | Pydub       | FFmpeg            |
    | -------------- | ----------- | ----------------- |
    | 10 GB dataset  | slower      | much faster       |
    | 100 GB dataset | painful     | manageable        |
    | 1 TB dataset   | impractical | standard approach |

2. Important Difference in Chunk Accuracy
    Your pydub code:
        range(0, len(audio), chunk_length_ms)
    creates EXACT 10-second windows.

    FFmpeg segmenting may split slightly differently depending on:
    - codec
    - timestamps
    - keyframes
    For WAV: usually accurate.

| Step               | Tool       |
| ------------------ | ---------- |
| MP3 → WAV          | FFmpeg     |
| Resampling         | FFmpeg     |
| Chunking           | FFmpeg     |
| Silence trimming   | FFmpeg     |
| Feature extraction | torchaudio |
| Spectrograms       | torchaudio |
| Training           | PyTorch    |

Comparison between librosa and torchaudio

1. librosa depends on numba → llvmlite → LLVM
Problem of python version support. Due to dependency chain:
librosa
  ↓
numba
  ↓
llvmlite
  ↓
LLVM

This creates:
- macOS build issues
- Apple Silicon issues
- Python version compatibility issues
- CMake/LLVM headaches


2. torchaudio integrates directly with PyTorch. This avoid librosa dependency issues.
    You are likely training with:
    - PyTorch
    - Whisper
    - wav2vec2
    - streaming transformers
    - DAC/Mimi
    
    torchaudio tensors work directly with GPU training.
    Example:
    waveform, sr = torchaudio.load("audio.wav")
    returns PyTorch tensors immediately. No conversion needed.

3. Better for large-scale datasets
    librosa:
    - often loads into NumPy arrays
    - CPU-oriented
    - more research-focused

    torchaudio:
    - optimized tensor pipelines
    - streaming support
    - batched loading
    - distributed training compatibility

    torchaudio important for:
    - Common Voice scale
    - multilingual corpora
    - ASR foundation models

4. FFmpeg backend support
    torchaudio can use FFmpeg backend internally.
    This gives:
    - fast decoding
    - streaming
    - better codec support
    - Much more production-friendly.
    
5. GPU acceleration potential
    librosa operations are mostly CPU-only.

    torchaudio integrates with:
    - CUDA
    - MPS (Apple GPU)

    PyTorch accelerators
    For:
    - spectrograms
    - augmentations
    - transforms

6. Better for realtime/streaming audio
    - WebSockets
    - streaming transformers
    - realtime speech systems
    - librosa is not ideal for streaming pipelines.
    - torchaudio is much better suited.

7. Production deployment
    Large speech systems typically use:
    - Component	Common Choice
    - Audio decoding	FFmpeg
    - Tensor audio ops	torchaudio
    - Training	PyTorch
    - Streaming	WebRTC/FFmpeg
    - Feature extraction	torchaudio

    librosa is less common in production ASR stacks now.

8. Apple Silicon compatibility
    librosa often causes:
    - wheel build issues
    - LLVM problems
    - numba issues
    on:
    M1
    M2
    M3 Macs
    torchaudio is generally smoother.

9. Memory efficiency
    - torchaudio supports:
    - streaming reads
    - partial reads
    - tensor pipelines
    - Better for huge datasets.

10. Better alignment with modern speech models
    Modern speech stacks:
        - Whisper
        - wav2vec2
        - NeMo
        - SeamlessM4T
        - speech tokenizers
        - Mimi/DAC pipelines
        are much closer to:
        - PyTorch tensors
        - torchaudio transforms
        than librosa.
        
        When librosa is still useful
        It is still excellent for:
            - MFCC visualization
            - chroma features
            - beat tracking
            - music analysis
            - DSP education
            - prototyping

Example:
import librosa
y, sr = librosa.load("audio.wav")
Very convenient and beginner-friendly.

Recommended Stack for Your Project
For your Marathi speech foundation model:
    |Task |	Recommended Tool|
    |Audio| conversion	FFmpeg|
    |Chunking|	FFmpeg|
    |Dataset loading	|torchaudio|
    |Spectrograms	|torchaudio|
    |Feature extraction	|torchaudio|
    |Training	|PyTorch|
    |Streaming	|WebSockets + FFmpeg|
    |Tokenization	|DAC/Mimi|
    |Metadata	|pandas|

    This avoids almost all LLVM/macOS dependency pain while scaling better later.

Chunking strategy for audio files
For DAC/Mimi-style speech tokenizers, you generally want:
short natural speech spans
not hard-cut phonemes
clean mono WAV
fixed sample rate
silence-aware preprocessing

Typical target:
2–10 second chunks
16kHz or 24kHz depending on tokenizer
For DAC/Mimi-style preprocessing with FFmpeg, a strong default is:

This produces:
- mono audio
- 24kHz
- silence-trimmed
- ~5 sec chunks

PCM WAV
Good for:
- DAC tokenizers
- Mimi-style audio tokenization
- streaming speech transformers
- speech-language models

Why 24kHz?
    Many modern speech tokenizers prefer:
    24kHz because it preserves:
    - more prosody
    - speaker identity
    - high-frequency detail
    - compared to 16kHz ASR-focused audio

Better Strategy Than Pure Fixed Chunking
For highest quality:
    - silence-aware splitting
    - then 5-second max windows
    instead of:
        - blindly every 1 second
    This reduces:
        - clipped syllables
        - broken phonemes
        - unnatural transitions


| Use Case          | Sample Rate  |
| ----------------- | ------------ |
| Whisper ASR       | 16kHz        |
| wav2vec2          | 16kHz        |
| DAC               | 24kHz common |
| Mimi              | 24kHz common |
| High-fidelity TTS | 24kHz–48kHz  |

'''

import os
import glob
import subprocess
from tqdm import tqdm

INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/chunks_by_silence" # And other porperties

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = glob.glob(f"{INPUT_DIR}/*.wav")

for filepath in tqdm(files):

    filename = os.path.splitext(os.path.basename(filepath))[0]

    out_pattern = os.path.join(
        OUTPUT_DIR,
        f"{filename}_%03d.wav"
    )

    '''```command = [
        "ffmpeg",
        "-i", filepath,
        "-f", "segment",
        "-segment_time", "1",
        "-c", "copy",
        out_pattern,
        "-y"
    ]```'''
    
    '''command = [
        "ffmpeg",
        "-i", filepath,
        "-f", "segment",
        "-segment_time", "1",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        out_pattern,
        "-y"
    ]'''
    command = [
        "ffmpeg",
        "-i", filepath,

        # resample + mono
        "-ar", "24000",
        "-ac", "1",

        # remove long silence
        "-af",
        "silenceremove=stop_periods=-1:stop_duration=0.4:stop_threshold=-40dB",

        # chunk into manageable spans
        "-f", "segment",
        "-segment_time", "8",

        # wav pcm
        "-c:a", "pcm_s16le",

        out_pattern,
        "-y"
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )