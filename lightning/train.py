import torch
import torchvision.datasets as datasets  # Standard datasets
from tqdm import tqdm
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import WhisperFinetuning
from dataset import WhisperDataset
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)

torch.set_float32_matmul_precision("medium")

# Hyperparameters
lr = 1e-5
batch_size = 16  # Reduced batch size
num_workers = 2  # Reduced number of workers
max_epochs = 1000
seed = 42  # Set your desired seed here

# Set seeds for reproducibility
set_seed(seed)

# Initialize model and data module
model = WhisperFinetuning(lr)
dm = WhisperDataset(data_dir="/Users/jon/code/aisg/deep-skilling-phase/pytorch-lightning-tutorial/data", batch_size=batch_size, num_workers=num_workers)

if __name__ == "__main__":
    # Initialize TensorBoard and Wandb loggers
    tb_logger = TensorBoardLogger("lightning_logs", name="whisper_finetuning")
    wandb_logger = WandbLogger(project="whisper_finetuning", log_model="all")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="mps",
        devices=[0],
        precision=16,
        logger=[tb_logger, wandb_logger],  # Add both loggers
        log_every_n_steps=1,
        deterministic=True,  # Ensure deterministic behavior
        # auto_lr_find=True,  # Add automatic learning rate finder
    )

    try:
        # Run test before training
        trainer.test(model, datamodule=dm)

        # Train the model
        trainer.fit(model, dm)

        # Run test after training
        trainer.test(model, datamodule=dm)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
