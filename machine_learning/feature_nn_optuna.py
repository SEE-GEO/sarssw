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

def objective(trial: optuna.trial.Trial) -> float:
    
    # Suggest a learning rate
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    
    # Suggest a dropout rate
    dropout_p = trial.suggest_float("dropout_p", 0, 1)
    
    train_dataset = sml.CustomDatasetFeatures(
        args.data_dir, 
        args.dataframe_path, 
        split='train', 
        scale_features=True, 
    )
    
    val_dataset = sml.CustomDatasetFeatures(
        args.data_dir, 
        args.dataframe_path, 
        split='val', 
        scale_features=True, 
        feature_mean=train_dataset.feature_mean, 
        feature_std=train_dataset.feature_std, 
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
    
    feature_dim = train_dataset.feature_dim
    model = sml.FeatureRegressor(
        feature_dim=feature_dim, 
        learning_rate=learning_rate, 
        mean_wave=train_dataset.mean_wave, 
        mean_wind=train_dataset.mean_wind,
        dropout_p=dropout_p,
        feature_mean=train_dataset.feature_mean, 
        feature_std=train_dataset.feature_std,
    )
    
    logger = pl.loggers.TensorBoardLogger("feat_final", name=f"lr={learning_rate}, dr={dropout_p}")
    
    trainer = pl.Trainer(
        logger=logger,
        enable_checkpointing=True,
        max_epochs=50,
        accelerator="gpu",
        devices=args.gpus,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
        deterministic=True,
    )

    hyperparameters = dict(learning_rate=learning_rate)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, train_loader, val_loader)
    return trainer.callback_metrics["val_loss"].item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True, help='The path to the data on alvis, most likely something similar to $TMPDIR/data in the slurm script')
    parser.add_argument('--dataframe_path', required=True, help='The path to the dataframe containing the metadata, features and labels for the data.')
    parser.add_argument('--gpus', type=int, required=True, help='Specify the number of GPUs to use for training. They should be requested in the slurm scrips')
    parser.add_argument('--checkpoint', help='Optional. The path to a checkpoint to restart from.')
    args = parser.parse_args()

    print('python version: ', sys.version)
    print('optuna version: ', optuna.__version__)
    print('pytorch lightning version: ', pl.__version__)

    if version.parse(pl.__version__) < version.parse("1.6.0"):
        raise RuntimeError("PyTorch Lightning>=1.6.0 is required for this example.")

    pl.seed_everything(0, workers=True)
    
    # Create the Optuna study and optimize the objective function
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100, timeout=2 * 3600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))