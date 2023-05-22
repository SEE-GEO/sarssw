import pickle
import zipfile
import os
import shutil
from itertools import islice

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
import pytorch_lightning as pl
from torchvision.models import resnet18
print("CUDA reachable?", torch.cuda.is_available())
print("PyTorch version?", torch.__version__)
print('lightning version', pl.__version__)

sar_dir = '/mimer/NOBACKUP/priv/chair/sarssw/sar_dataset'
fl_df_path = '/mimer/NOBACKUP/priv/chair/sarssw/sar_dataset_features_labels_18_april/sar_dataset.pickle'

n_files = 10000
file_filter = lambda fn: fn.is_file() and fn.name.endswith('.nc') and 'IW' in fn.name
sar_dataset_files = list(islice((fn.name for fn in os.scandir(sar_dir) if file_filter(fn)), n_files))

with open(fl_df_path, 'rb') as f:
    fl_df = pickle.load(f)

fl_df = fl_df[fl_df.file_name.isin(sar_dataset_files)]

class CustomDataset(Dataset):
  def __init__(self, dataset_df, sar_dir):
    # preprocess with hom test, feature extract and everything
    #filter homogenious images with IW mode and no na
    hom_df = dataset_df[dataset_df.hom_test]
    hom_df = hom_df.dropna()

    #merge the vv and vh polarization
    VV_df, VH_df = hom_df[hom_df.pol == 'VV'], hom_df[hom_df.pol == 'VH']
    merge_df = VV_df.merge(VH_df, on='file_name', suffixes=('_VV', '_VH'))

    self.features = [
        'sigma_mean', 'sigma_var', 'sigma_mean_over_var', 
        'sigma_min', 'sigma_max', 'sigma_range'
    ]

    self.df = merge_df
    self.sar_dir = sar_dir

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    df_row = self.df.iloc[index]
    # Get image file name from the DataFrame
    file_name = df_row.file_name
    image_path = os.path.join(self.sar_dir, file_name)
    image = xr.open_dataset(image_path).sigma0.values.astype(np.float32)

    # Get features and labels from the DataFrame
    features = df_row[[f + p for f in self.features for p in ['_VV', '_VH']]].values.astype(np.float32)
    labels = df_row[['SWH_value_VV', 'WSPD_value_VV']].values.astype(np.float32)

    return torch.tensor(image), torch.tensor(features), torch.tensor(labels)

class ImageFeatureRegressor(pl.LightningModule):
    def __init__(self, feature_dim):
        super(ImageFeatureRegressor, self).__init__()

        self.image_cnn = resnet18(pretrained=False)
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

        self.log("train_mse", mse_loss, prog_bar=True)
        self.log("train_rmse", rmse, prog_bar=True)
        self.log("train_mae", mae, prog_bar=True)

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

        # Return the metric(s) as a dictionary
        return {'val_mse': val_mse_loss, 'val_rmse': val_rmse, 'val_mae': val_mae}

    def validation_epoch_end(self, outputs):
        avg_val_mse = torch.stack([x['val_mse'] for x in outputs]).mean()
        avg_val_rmse = torch.stack([x['val_rmse'] for x in outputs]).mean()
        avg_val_mae = torch.stack([x['val_mae'] for x in outputs]).mean()

        self.log("avg_val_mse", avg_val_mse)
        self.log("avg_val_rmse", avg_val_rmse)
        self.log("avg_val_mae", avg_val_mae)

        self.final_val_mse = avg_val_mse.item()
        self.final_val_rmse = avg_val_rmse.item()
        self.final_val_mae = avg_val_mae.item()

        return {'avg_val_mse': avg_val_mse, 'avg_val_rmse': avg_val_rmse, 'avg_val_mae': avg_val_mae}


# Create the dataset
dataset = CustomDataset(fl_df, sar_dir)
train_size, val_size = 0.7, 0.3
# Split the dataset into training and validation sets
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders for training and validation sets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=16)

# Create the LightningModule and Trainer instances
feature_dim = len(dataset[0][1])
model = ImageFeatureRegressor(feature_dim)
trainer = pl.Trainer(accelerator='gpu', max_epochs=50)

# Train the model
trainer.fit(model, train_loader, val_loader)

# Access and print the final validation metrics
print("Final Validation MSE:", model.final_val_mse)
print("Final Validation RMSE:", model.final_val_rmse)
print("Final Validation MAE:", model.final_val_mae)