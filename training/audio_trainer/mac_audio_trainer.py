import time
import torch
from torch.utils.data import DataLoader
from dac_token_dataset import DACTokenDataset
from audio_transformer import AudioTransformer
from training_checkpoint import load_checkpoint, save_checkpoint, save_bestmodel
from collate_dac_tokens import collate_fn

MAX_RUNTIME_SECONDS = 300 # 5 min 900# 15 min  #3600 #1 hour
CHECKPOINT_INTERVAL = 20

DEVICE = "cpu"  #"cuda"

dataset = DACTokenDataset(
    #"data/tokens",
    "data/training_dry_run/tokens",
    limit=50
)
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
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

start_epoch, global_step, best_loss = (
    load_checkpoint(
        model,
        optimizer
    )
)

start_time = time.time()

epoch = start_epoch

while True:

    for x, y in loader:

        if time.time() - start_time > MAX_RUNTIME_SECONDS:

            print("Runtime limit reached")

            save_checkpoint(
                model,
                optimizer,
                epoch,
                global_step,
                best_loss
            )

            raise SystemExit

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        loss_value = loss.item()

        recent_losses.append(loss_value)

        running_loss += loss_value

        if len(recent_losses) > 100:
            running_loss -= recent_losses.pop(0)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        optimizer.step()

        global_step += 1

        if global_step % CHECKPOINT_INTERVAL == 0:

            avg_loss = (
                running_loss
                / len(recent_losses)
            )

            if avg_loss < best_loss:
                best_loss = avg_loss

                save_bestmodel(
                    global_step,
                    model,
                    epoch,
                    best_loss
                )

                print(
                    f"New best model "
                    f"avg_loss={avg_loss:.4f}"
                )
    
            save_checkpoint(
                model,
                optimizer,
                epoch,
                global_step,
                best_loss
            )

            print(
                f"step={global_step} "
                f"loss={loss_value:.4f} "
                f"avg_loss={avg_loss:.4f}"
            )

    epoch += 1