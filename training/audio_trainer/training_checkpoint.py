import os
import torch


#CHECKPOINT_DIR = "checkpoints"

CHECKPOINT_FILE = "checkpoint.pt"
BEST_MODEL_FILE = "best_model.pt"
'''os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)'''

'''BEST_MODEL_FILE = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pt"
)'''

'''CHECKPOINT_FILE = os.path.join(
    CHECKPOINT_DIR,
    "checkpoint.pt"
)'''

import os
import torch

def save_checkpoint(
    checkpoint_dir,
    model,
    optimizer,
    epoch,
    global_step,
    best_loss,
    shard_idx,
    batch_idx
):

    checkpoint_file = os.path.join(
        checkpoint_dir,
        CHECKPOINT_FILE
    )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": best_loss,
            "shard_idx": shard_idx,
            "batch_idx": batch_idx
        },
        checkpoint_file
    )

def save_bestmodel(
        checkpoint_dir,
        global_step,
        model,
        epoch,
        best_loss
    ):
    bestmodel_file = os.path.join(
        checkpoint_dir,
        BEST_MODEL_FILE
    )
    torch.save(
        {
            "global_step": global_step,
            "epoch": epoch,
            "best_loss": best_loss,
            "model": model.state_dict()
        },
        bestmodel_file
    )

def load_checkpoint(
    checkpoint_dir,
    model,
    optimizer
):
    checkpoint_file = os.path.join(
        checkpoint_dir,
        CHECKPOINT_FILE
    )

    try:

        ckpt = torch.load(
            checkpoint_file,
            map_location="cpu"
        )

        model.load_state_dict(
            ckpt["model"]
        )

        optimizer.load_state_dict(
            ckpt["optimizer"]
        )

        return (
            ckpt["epoch"],
            ckpt["global_step"],
            ckpt["best_loss"],
            ckpt.get("shard_idx", 0),
            ckpt.get("batch_idx", 0)
        )

    except FileNotFoundError:

        return (
            0,      # epoch
            0,      # global_step
            float("inf"),
            0,      # shard_idx
            0       # batch_idx
        )
