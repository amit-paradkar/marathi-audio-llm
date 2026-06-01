'''
Create shards meaning aggregate say 1000.pt files into a single shard file. This
is useful for training on colab as management of 20K files is difficult.
'''
import glob
import os
import torch
from tqdm import tqdm

TOKEN_DIR = "data/tokens_by_silence"
SHARD_DIR = "data/shards_by_silence"

#TOKEN_DIR = "data/training_dry_run/tokens"
#SHARD_DIR = "data/training_dry_run/shards"

os.makedirs(SHARD_DIR, exist_ok=True)

SHARD_SIZE = 1000

files = sorted(
    glob.glob(
        f"{TOKEN_DIR}/*.pt"
    )
)

print(
    f"Found {len(files)} files"
)
if len(files) == 0:
    f"Found {len(files)} files"
    raise SystemExit

for shard_idx in tqdm(
    range(
        0,
        len(files),
        SHARD_SIZE
    )
):

    shard_files = files[
        shard_idx:
        shard_idx + SHARD_SIZE
    ]

    shard = []

    for filepath in shard_files:

        tokens = torch.load(
            filepath,
            map_location="cpu"
        )

        shard.append(tokens)

    output_path = (
        f"{SHARD_DIR}/"
        f"shard_{shard_idx // SHARD_SIZE:03d}.pt"
    )

    torch.save(
        shard,
        output_path
    )

    print(
        f"Saved {output_path}"
    )