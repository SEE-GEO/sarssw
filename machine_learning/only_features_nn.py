import pickle
import zipfile
import os
import shutil
from itertools import islice

import numpy as np
import pandas as pd
import xarray as xr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import LearningRateFinder
import torch.nn.functional as F
import pytorch_lightning as pl
print("CUDA reachable?", torch.cuda.is_available())
print("PyTorch version?", torch.__version__)
print('lightning version', pl.__version__)

torch.set_float32_matmul_precision('high')

sar_dir = '/mimer/NOBACKUP/priv/chair/sarssw/sar_dataset'
fl_df_path = '/mimer/NOBACKUP/priv/chair/sarssw/sar_dataset_features_labels_27_april/sar_dataset_split.pickle'

#n_files = 10000
#file_filter = lambda fn: fn.is_file() and fn.name.endswith('.nc') and 'IW' in fn.name
#sar_dataset_files = list(islice((fn.name for fn in os.scandir(sar_dir) if file_filter(fn)), n_files))
sar_dataset_files = [f for f in os.listdir(sar_dir) if f.endswith('.nc') and 'IW' in f]

with open(fl_df_path, 'rb') as f:
    fl_df = pickle.load(f)

fl_df = fl_df[fl_df.file_name.isin(sar_dataset_files)]

class CustomDataset(Dataset):
    def __init__(self, dataset_df, split='train', base_features=None, scale_features=True, mean=None, std=None):
        # preprocess with hom test, feature extract and everything
        #filter homogenious images with IW mode and no na
        split_df = dataset_df[dataset_df.split == split]
        hom_df = split_df[split_df.hom_test]
        
        db_feats = [
            'sigma_mean','sigma_var', 'sigma_mean_over_var', 
        ]

        for feat in db_feats:
            hom_df[feat + '_dB'] = 10 * np.log10(hom_df[feat])
        
        hom_df = hom_df.dropna()

        #merge the vv and vh polarization
        VV_df, VH_df = hom_df[hom_df.pol == 'VV'], hom_df[hom_df.pol == 'VH']
        merge_df = VV_df.merge(VH_df, on='file_name', suffixes=('_VV', '_VH'))

        if base_features is None:
            # base_features = [
            #     'sigma_mean', 'sigma_var', 'sigma_mean_over_var', 
            #     'sigma_min', 'sigma_max', 'sigma_range'
            #]
            
            base_features = [
                'contrast', 'dissimilarity', 'homogeneity', 
                'energy', 'correlation', 'ASM', 'sigma_mean',
                'sigma_var', 'sigma_mean_over_var', 'sigma_min', 
                'sigma_max', 'sigma_range'
            ] + [feat + '_dB' for feat in db_feats]
        
        self.features = [f + p for f in base_features for p in ['_VV', '_VH']]
        self.feature_dim = len(self.features) 
        features_array = merge_df[self.features].values.astype(np.float32)
        features_tensor = torch.from_numpy(features_array)

        if scale_features:
            # If the dataset is the training set, fit and transform the features
            if split == 'train':
                self.mean = features_tensor.mean(dim=0)
                self.std = features_tensor.std(dim=0)
                features_tensor = (features_tensor - self.mean) / self.std
            # If the dataset is the validation or test set, transform the features using the mean and std computed on the training set
            elif mean is not None and std is not None:
                self.mean = mean
                self.std = std
                features_tensor = (features_tensor - self.mean) / self.std

        self.features_tensor = features_tensor      
        
        self.labels_cols = ['SWH_value_VV'] # TODO add back wind speed, removed for testing
        labels_array = merge_df[self.labels_cols].values.astype(np.float32)
        self.labels_tensor = torch.from_numpy(labels_array)

    def __len__(self):
        return len(self.features_tensor)

    def __getitem__(self, index):
        features = self.features_tensor[index]
        labels = self.labels_tensor[index]

        return features, labels

class FeatureRegressor(pl.LightningModule):
    def __init__(self, feature_dim, learning_rate=1e-3):
        super(FeatureRegressor, self).__init__()
        self.learning_rate = learning_rate
        self.fc1 = nn.Linear(feature_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1) # TODO, add back wind speed, removed for testing
     
    def forward(self, features):
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

    def training_step(self, batch, batch_idx):
        feature_batch, target_batch = batch
        predictions = self(feature_batch)

        loss = nn.MSELoss()
        mse_loss = loss(predictions, target_batch)

        rmse = torch.sqrt(mse_loss)
        mae = torch.mean(torch.abs(predictions - target_batch))

        self.log("train_mse", mse_loss, prog_bar=True)
        self.log("train_rmse", rmse, prog_bar=True)
        self.log("train_mae", mae, prog_bar=True)

        return mse_loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def validation_step(self, batch, batch_idx):
        feature_batch, target_batch = batch
        predictions = self(feature_batch)

        # Compute your evaluation metric(s)
        loss = nn.MSELoss()
        val_mse_loss = loss(predictions, target_batch)

        val_rmse = torch.sqrt(val_mse_loss)
        val_mae = torch.mean(torch.abs(predictions - target_batch))

        self.log("val_mse_loss", val_mse_loss, prog_bar=True)
        self.log("val_rmse", val_rmse, prog_bar=True)
        self.log("val_mae", val_mae, prog_bar=True)

        # Return the metric(s) as a dictionary
        return {'val_mse': val_mse_loss, 'val_rmse': val_rmse, 'val_mae': val_mae}

# Create data loaders for training and validation sets
train_dataset = CustomDataset(fl_df, split='train', scale_features=True)
val_dataset = CustomDataset(fl_df, split='val', scale_features=True, mean=train_dataset.mean, std=train_dataset.std)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=16)

# Create the LightningModule and Trainer instances
feature_dim = train_dataset.feature_dim
model = FeatureRegressor(feature_dim)

class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)

trainer = pl.Trainer(accelerator='gpu', max_epochs=100, callbacks=[FineTuneLearningRateFinder(milestones=(5, 10))], log_every_n_steps=30)

# Train the model
trainer.fit(model, train_loader, val_loader)