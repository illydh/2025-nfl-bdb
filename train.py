import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import warnings

from dataset import get_dataloader, MaskedPlayerDataset
from models import GhostFormerLitModel

warnings.filterwarnings("ignore")

def main():
    pl.seed_everything(42)

    # We use part of train frames for training, part for val.
    # Since we saved everything together in train_frames.pt, let's load it.
    print("Loading datasets...")
    full_dataset = MaskedPlayerDataset("data/train_frames.pt")
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    model = GhostFormerLitModel(learning_rate=3e-4)

    logger = TensorBoardLogger("tb_logs", name="ghost_former")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="ghost_former-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")

    trainer = pl.Trainer(
        max_epochs=1,  # Short max epochs for quick test, set higher in practice
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        devices=1,
        log_every_n_steps=50,
        val_check_interval=0.5
    )

    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
