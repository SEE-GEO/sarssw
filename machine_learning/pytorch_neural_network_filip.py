#!/usr/bin/env python3
"""
This is a script for training a resnet CNN model on images with 2 polarisations.
"""

import pickle
import zipfile
import os
import shutil
from itertools import islice
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor
from torchvision.models import resnet18
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class RandomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return TF.rotate(img, angle)
    
class RandomHFlipTransform:
    """Randomly horizontally flip the image with specified probability."""

    def __init__(self, probability):
        self.probability = probability

    def __call__(self, img):
        if random.random() < self.probability:
            return TF.hflip(img)
        else:
            return img

class RandomVFlipTransform:
    """Randomly vertically flip the image with specified probability."""

    def __init__(self, probability):
        self.probability = probability

    def __call__(self, img):
        if random.random() < self.probability:
            return TF.vflip(img)
        else:
            return img

class CustomDataset(Dataset):
  def __init__(self, data_dir, dataframe_path, transforms=None):
    if not os.path.isdir(data_dir):
       raise ValueError(f"The data directory {data_dir} not found")
    
    sar_dataset_files = [dir_entry.name for dir_entry in os.scandir(data_dir)]

    with open(dataframe_path, 'rb') as f:
        fl_df = pickle.load(f)

    #Filter for only images in the data_dir
    fl_df = fl_df[fl_df.file_name.isin(sar_dataset_files)]

    #merge the vv and vh polarization
    VV_df, VH_df = fl_df[fl_df.pol == 'VV'], fl_df[fl_df.pol == 'VH']
    fl_df_merged = VV_df.merge(VH_df, on='file_name', suffixes=('_VV', '_VH'))

    self.features = [
        'sigma_mean', 'sigma_var', 'sigma_mean_over_var', 
        'sigma_min', 'sigma_max', 'sigma_range'
    ]

    self.df = fl_df_merged
    self.sar_dir = data_dir
    if transforms is None:
        self.transforms = Compose([transforms])

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    df_row = self.df.iloc[index]
    # Get image file name from the DataFrame
    file_name = df_row.file_name
    image_path = os.path.join(self.sar_dir, file_name)
    image = torch.tensor(xr.open_dataset(image_path).sigma0.values.astype(np.float32)) #TODO should we use 64 bit?

    # Apply transform if it exists
    if self.transform is not None:
        image = self.transform(image)

    # Get features and labels from the DataFrame
    features = df_row[[f + p for f in self.features for p in ['_VV', '_VH']]].values.astype(np.float32)
    #The labels are the same for the two polarisations to we pick the labels from VV
    labels = df_row[['SWH_value_VV', 'WSPD_value_VV']].values.astype(np.float32)

    return image, torch.tensor(features), torch.tensor(labels)

class ImageFeatureRegressor(pl.LightningModule):
    def __init__(self, feature_dim):
        super(ImageFeatureRegressor, self).__init__()
        self.save_hyperparameters()

        self.image_cnn = resnet18() #Default to no pre-training
        self.image_cnn.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.image_cnn.fc = nn.Identity()

        self.fc1 = nn.Linear(512 + feature_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        
        self.final_val_mse = None
        self.final_val_rmse = None
        self.final_val_mae = None
        
    def forward(self, image, features):
        image_output = self.image_cnn(image)
        image_output = image_output.view(image_output.size(0), -1)
        
        combined = torch.cat((image_output, features), dim=1)
        
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

    def training_step(self, batch, batch_idx):
        image_batch, feature_batch, target_batch = batch
        predictions = self(image_batch, feature_batch)

        loss = nn.MSELoss()
        mse_loss = loss(predictions, target_batch)

        rmse = torch.sqrt(mse_loss)
        mae = torch.mean(torch.abs(predictions - target_batch))

        log_dict = {"train_mse":mse_loss, "train_rmse":rmse, "train_mae":mae}
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)

        return mse_loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
    
    def validation_step(self, batch, batch_idx):
        image_batch, feature_batch, target_batch = batch
        predictions = self(image_batch, feature_batch)

        # Compute your evaluation metric(s)
        loss = nn.MSELoss()
        val_mse_loss = loss(predictions, target_batch)

        val_rmse = torch.sqrt(val_mse_loss)
        val_mae = torch.mean(torch.abs(predictions - target_batch))

        log_dict = {"val_mse":val_mse_loss, "val_rmse":val_rmse, "val_mae":val_mae}

        self.log_dict(log_dict, prog_bar=True)

        # Return the metric(s) as a dictionary
        return log_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True, help='The path to the data on alvis, most likely something similar to $TMPDIR/data in the slurm script')
    parser.add_argument('--dataframe_path', required=True, help='The path to the dataframe containing the metadata, features and labels for the data.')
    parser.add_argument('--checkpoint', help='Optional. The path to a checkpoint to restart from.')
    args = parser.parse_args()

    print("CUDA reachable?", torch.cuda.is_available())
    print("PyTorch version?", torch.__version__)
    print('lightning version', pl.__version__)

    # Create the datasets
    train_dataset = CustomDataset(
        os.path.join(args.data_dir, 'train'),
        args.dataframe_path,
        transforms=[
            RandomRotationTransform(angles=[0, 90, 180, 270]),
            RandomHFlipTransform(probability=0.5),
            RandomVFlipTransform(probability=0.5)]
            )
    val_dataset = CustomDataset(os.path.join(args.data_dir, 'val'), args.dataframe_path)

    # Create data loaders for training and validation sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)

    # Program callbacks
    # saves top-2 checkpoints based on "val_mse" metric
    checkpoint_callback_best = ModelCheckpoint(
        save_top_k=2,
        monitor="val_mse",
        mode="min",
        save_on_train_epoch_end=True,
        filename="best_val_loss-{epoch:02d}-{val_mse:.2f}",
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
       monitor="val_mse",
       min_delta=0.00,
       patience=5,
       verbose=False,
       mode="min"
    )

    # Create the LightningModule and Trainer instances
    feature_dim = len(train_dataset[0][1])
    model = ImageFeatureRegressor(feature_dim)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=1000, devices=2, callbacks=[checkpoint_callback_best, checkpoint_callback_latest, early_stop_callback])

    # Train the model
    # No checkpoint provided, so we train from scratch
    if args.checkpoint is None:
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.checkpoint)