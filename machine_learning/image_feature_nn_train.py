import pickle
import zipfile
import os
import shutil
from itertools import islice
from packaging import version
import sys
from functools import lru_cache
import argparse
import random

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, resnet34, resnet50, vgg16, vgg19, resnet, vgg
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, RandomVerticalFlip, ToTensor
import torchvision.transforms.functional as TF

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import optuna
from optuna.integration import PyTorchLightningPruningCallback

import sarssw_ml_lib as sml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True, help='The path to the data on alvis, most likely something similar to $TMPDIR/data in the slurm script')
    parser.add_argument('--dataframe_path', required=True, help='The path to the dataframe containing the metadata, features and labels for the data.')
    parser.add_argument('--gpus', type=int, required=True, help='Specify the number of GPUs to use for training. They should be requested in the slurm scrips')
    args = parser.parse_args()

    print('python version: ', sys.version)
    print('optuna version: ', optuna.__version__)
    print('pytorch lightning version: ', pl.__version__)

    if version.parse(pl.__version__) < version.parse("1.6.0"):
        raise RuntimeError("PyTorch Lightning>=1.6.0 is required for this example.")

    pl.seed_everything(0, workers=True)
    
    # Calculate mean and standard deviation of the training set
    train_dataset_norm = sml.CustomDataset(
        args.data_dir,
        args.dataframe_path
        )

    pixel_mean, pixel_std = sml.dataset_mean_std(train_dataset_norm)
    
    # output
    print('pixel mean: '  + str(pixel_mean))
    print('pixel std:  '  + str(pixel_std))
    
    train_transform = Compose([
                Normalize(mean=pixel_mean, std=pixel_std),
                sml.RandomRotationTransform(angles=[0, 90, 180, 270]),
                RandomHorizontalFlip(0.5),
                RandomVerticalFlip(0.5),
            ])

    val_transform = Normalize(mean=pixel_mean, std=pixel_std)
    
    # Suggest a learning rate
    learning_rate = 0.0005
    
    # Suggest a dropout rate
    dropout_p = 0.2
    
    # Suggest if pretrained or not
    pretrained = True
    
    # Suggest a pretrained model name
    model_name = 'resnet50'

    train_dataset = sml.CustomDataset(
        args.data_dir, 
        args.dataframe_path, 
        split='train', 
        scale_features=True, 
        transform=train_transform
    )
    
    val_dataset = sml.CustomDataset(
        args.data_dir, 
        args.dataframe_path, 
        split='val', 
        scale_features=True, 
        feature_mean=train_dataset.feature_mean, 
        feature_std=train_dataset.feature_std, 
        transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
    
    feature_dim = train_dataset.feature_dim
    model = sml.ImageFeatureRegressor(
        model_name=model_name,
        pretrained=pretrained,
        feature_dim=feature_dim, 
        learning_rate=learning_rate, 
        mean_wave=train_dataset.mean_wave, 
        mean_wind=train_dataset.mean_wind,
        dropout_p=dropout_p,
        feature_mean=train_dataset.feature_mean, 
        feature_std=train_dataset.feature_std,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
    )
    
    logger = pl.loggers.TensorBoardLogger("final_training_logger_new_loss", name=f"lr={learning_rate}, dr={dropout_p}, model={model_name}, pre={pretrained}")
    
    # Program callbacks
    # saves top-2 checkpoints based on "val_loss" metric
    checkpoint_callback_best = ModelCheckpoint(
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        save_on_train_epoch_end=True,
        filename="best_val_loss-{epoch:02d}-{val_loss:.2f}",
    )

    # saves last-2 checkpoints based on epoch
    checkpoint_callback_latest = ModelCheckpoint(
        save_top_k=2,
        monitor="epoch",
        mode="max",
        save_on_train_epoch_end=True,
        filename="latest-epoch-{epoch:02d}",
    )

    early_stop_callback = EarlyStopping(
       monitor="val_loss",
       min_delta=0.00,
       patience=5,
       verbose=False,
       mode="min"
    )

    trainer = pl.Trainer(
        logger=logger,
        enable_checkpointing=True,
        max_epochs=100,
        accelerator="gpu",
        devices=args.gpus,
        callbacks=[checkpoint_callback_best, checkpoint_callback_latest],
    )

    hyperparameters = dict(learning_rate=learning_rate)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, train_loader, val_loader)