'''
uv run python training/audio_trainer/audio_trainer_1shard_1time.py \
    --shard_dir data/training_dry_run/shards \
    --checkpoint_dir data/training_dry_run/checkpoints

Colab
!python training/audio_trainer/audio_trainer_1shard_1time.py \
    --shard_dir /content/drive/MyDrive/marathi/shards \
    --checkpoint_dir /content/drive/MyDrive/marathi/checkpoints

'''

import glob
import time
import torch
import os
from torch.utils.data import DataLoader

from dac_shard_dataset import ShardDataset
from audio_transformer import AudioTransformer
from training_checkpoint import (
    load_checkpoint,
    save_checkpoint,
    save_bestmodel
)
from collate_dac_tokens import collate_fn
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "--shard_dir",
    type=str,
    required=True,
    help="Directory containing shard files"
)

parser.add_argument(
    "--checkpoint_dir",
    type=str,
    required=True,
    help="Directory for checkpoints and best models"
)

args = parser.parse_args()

SHARD_DIR = args.shard_dir
CHECKPOINT_DIR = args.checkpoint_dir

print(f"Using shard directory: {SHARD_DIR}")
print(f"Using checkpoint directory: {CHECKPOINT_DIR}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device identified: {DEVICE}")

if DEVICE == 'cuda':
    MAX_RUNTIME_SECONDS = 300
    CHECKPOINT_INTERVAL = 20
    BATCH_SIZE = 4
else:
    MAX_RUNTIME_SECONDS = 60
    CHECKPOINT_INTERVAL = 10
    BATCH_SIZE = 2

print(f"Max runtime (seconds): {MAX_RUNTIME_SECONDS}")
print(f"Checkpoint interval (seconds): {CHECKPOINT_INTERVAL}")
print(f"Batch size : {BATCH_SIZE}")

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

shard_files = sorted(
    glob.glob(
        f"{SHARD_DIR}/*.pt"
    )
)

model = AudioTransformer().to(
    DEVICE
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)

criterion = torch.nn.CrossEntropyLoss(
    ignore_index=-100
)

running_loss = 0.0
recent_losses = []

best_loss = float("inf")

(
    start_epoch,
    global_step,
    best_loss,
    start_shard_idx,
    start_batch_idx
) = load_checkpoint(
    CHECKPOINT_DIR,
    model,
    optimizer
)

start_time = time.time()

epoch = start_epoch

while True:

    for shard_idx in range(
        start_shard_idx,
        len(shard_files)
    ):

        shard_file = shard_files[
            shard_idx
        ]

        print(
            f"\nLoading shard "
            f"{shard_idx + 1}/"
            f"{len(shard_files)}"
        )

        shard = torch.load(
            shard_file,
            map_location="cpu"
        )

        dataset = ShardDataset(
            shard,
            seq_len=1024
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,   # IMPORTANT
            collate_fn=collate_fn
        )

        for batch_idx, (x, y) in enumerate(loader):

            #
            # Resume mid-shard
            #
            if (
                shard_idx == start_shard_idx
                and batch_idx < start_batch_idx
            ):
                continue

            #
            # Runtime limit reached
            #
            if (
                time.time()
                - start_time
                > MAX_RUNTIME_SECONDS
            ):

                print(
                    "Runtime limit reached"
                )

                save_checkpoint(
                    CHECKPOINT_DIR,
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    best_loss,
                    shard_idx,
                    batch_idx
                )

                raise SystemExit

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)

            loss = criterion(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),
                y.reshape(-1)
            )

            loss_value = loss.item()

            recent_losses.append(
                loss_value
            )

            running_loss += loss_value

            if len(recent_losses) > 100:

                running_loss -= (
                    recent_losses.pop(0)
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            #
            # Gradient clipping
            #
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            if global_step == 100:
                avg_loss = (
                    running_loss
                    / len(recent_losses)
                )

                print(
                    f"\n100-step check:"
                    f" avg_loss={avg_loss:.4f}"
                )
    
            if global_step % 10 == 0:
                avg_loss = (
                    running_loss
                    / len(recent_losses)
                )

                print(
                    f"step={global_step} "
                    f"loss={loss_value:.4f} "
                    f"avg_loss={avg_loss:.4f}"
                )
    
            global_step += 1

            #
            # Periodic checkpoint
            #
            if (
                global_step
                % CHECKPOINT_INTERVAL
                == 0
            ):

                avg_loss = (
                    running_loss
                    / max(
                        len(recent_losses),
                        1
                    )
                )

                if avg_loss < best_loss:

                    best_loss = avg_loss

                    save_bestmodel(
                        CHECKPOINT_DIR,
                        global_step,
                        model,
                        epoch,
                        best_loss
                    )

                    print(
                        "New best model "
                        f"avg_loss="
                        f"{avg_loss:.4f}"
                    )

                save_checkpoint(
                    CHECKPOINT_DIR,
                    model,
                    optimizer,
                    epoch,
                    global_step,
                    best_loss,
                    shard_idx,
                    batch_idx
                )

                print(
                    f"epoch={epoch} "
                    f"shard={shard_idx} "
                    f"batch={batch_idx} "
                    f"step={global_step} "
                    f"loss={loss_value:.4f} "
                    f"avg_loss={avg_loss:.4f}"
                )

        #
        # Finished entire shard
        #
        save_checkpoint(
            CHECKPOINT_DIR,
            model,
            optimizer,
            epoch,
            global_step,
            best_loss,
            shard_idx + 1,
            0
        )

        print(
            f"Finished shard "
            f"{shard_idx + 1}"
        )

        #
        # Reset batch position
        #
        start_batch_idx = 0

        #
        # Memory cleanup
        #
        del shard
        del dataset
        del loader

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(
        f"\nCompleted epoch {epoch}"
    )
    #
    # Finished entire epoch
    #
    save_checkpoint(
        CHECKPOINT_DIR,
        model,
        optimizer,
        epoch + 1,
        global_step,
        best_loss,
        0,
        0
    )

    epoch += 1

    start_shard_idx = 0
    start_batch_idx = 0
    
    print(
        f"\nStarted next epoch {epoch}"
    )
