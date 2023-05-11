import pickle
import zipfile
import os
import shutil
from itertools import islice
from packaging import version
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.models import resnet18
import pytorch_lightning as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback


class CustomDataset(Dataset):
    def __init__(self, dataset_df, sar_dir, split='train', base_features=None, scale_features=True, image_unit='lin', mean=None, std=None):
        # preprocess with hom test, feature extract and everything
        #filter homogenious images with IW mode and no na
        split_df = dataset_df[dataset_df.split == split]
        hom_df = split_df[split_df.hom_test].copy()
        
        db_feats = [
            'sigma_mean','sigma_var', 'sigma_mean_over_var', 
        ]

        for feat in db_feats:
            aa = hom_df[feat]
            hom_df.loc[:, feat + '_dB'] = np.log10(np.where(aa>0.0, aa, 1e-300))
        
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
                #'contrast', 'dissimilarity', 'homogeneity', 
                #'energy', 'correlation', 'ASM', 
                'sigma_mean',
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
        
        self.wave_col = 'SWH_value_VV'
        self.mean_wave = merge_df[self.wave_col].mean()
        print(self.mean_wave)
        self.wave_tensor = torch.from_numpy(merge_df[self.wave_col].values.astype(np.float32))
        
        self.wind_col = 'WSPD_value_VV'
        self.mean_wind = merge_df[self.wind_col].mean()
        print(self.mean_wind)
        self.wind_tensor = torch.from_numpy(merge_df[self.wind_col].values.astype(np.float32))
        
        self.sar_dir = sar_dir
        self.file_names = merge_df.file_name
        
        if image_unit.lower() in ['lin', 'db']:
            self.image_unit = image_unit.lower()
        else:
            raise ValueError(f"Unknown unit {image_unit}")

    def __len__(self):
        return len(self.features_tensor)

    def __getitem__(self, index):
        features = self.features_tensor[index]
        labels = self.wave_tensor[index], self.wind_tensor[index]
        
        file_name = self.file_names.iloc[index]
        image_path = os.path.join(self.sar_dir, file_name)
        sigma0 = xr.open_dataset(image_path).sigma0.values.astype(np.float32)
        if self.image_unit == 'db':
            sigma0 = np.log10(np.where(sigma0 > 0.0, sigma0, 1e-10))
        image = torch.tensor(sigma0)

        return image, features, labels

class CustomLoss(nn.Module):
    def __init__(self, mean_wind, mean_wave):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.mean_wind = mean_wind
        self.mean_wave = mean_wave

    def forward(self, output_wind, output_wave, target_wind, target_wave):
        mse_wind = self.mse(output_wind, target_wind)
        mse_wave = self.mse(output_wave, target_wave)

        # normalize the losses by dividing by the means
        mse_wind_normalized = mse_wind / self.mean_wind
        mse_wave_normalized = mse_wave / self.mean_wave

        # equally weighted root mean square of the losses
        loss = torch.sqrt((mse_wind_normalized + mse_wave_normalized) / 2)
        return loss

class ImageFeatureRegressor(pl.LightningModule):
    def __init__(self, feature_dim, learning_rate, mean_wind, mean_wave):
        super(ImageFeatureRegressor, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.loss_fn = CustomLoss(mean_wave=mean_wave, mean_wind=mean_wind)
        
        self.image_cnn = resnet18()  # Default to no pre-training
        self.image_cnn.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.image_cnn.fc = nn.Identity()

        self.fc1 = nn.Linear(512 + feature_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)

        self.fc4_wave = nn.Linear(1024, 256)
        self.fc5_wave = nn.Linear(256, 128)
        self.fc6_wave = nn.Linear(128, 1)

        self.fc4_wind = nn.Linear(1024, 256)
        self.fc5_wind = nn.Linear(256, 128)
        self.fc6_wind = nn.Linear(128, 1)
        
    def forward(self, image, features):
        image_output = self.image_cnn(image)
        image_output = image_output.view(image_output.size(0), -1)
        
        combined = torch.cat((image_output, features), dim=1)
        
        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        
        # branch wave
        x_wave = torch.relu(self.fc4_wave(x))
        x_wave = torch.relu(self.fc5_wave(x_wave))
        x_wave = self.fc6_wave(x_wave)
        
        # branch wind
        x_wind = torch.relu(self.fc4_wind(x))
        x_wind = torch.relu(self.fc5_wind(x_wind))
        x_wind = self.fc6_wind(x_wind)
        
        return x_wave, x_wind

    def training_step(self, batch, batch_idx):
        image_batch, feature_batch, (target_wind, target_wave) = batch
        output_wind, output_wave = self(image_batch, feature_batch)

        loss = self.loss_fn(
            output_wind=output_wind, 
            output_wave=output_wave, 
            target_wind=target_wind, 
            target_wave=target_wave
        )
        
        log_dict = {"train_loss":loss}
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def validation_step(self, batch, batch_idx):
        image_batch, feature_batch, (target_wind, target_wave) = batch
        predictions_wave, predictions_wind = self(image_batch, feature_batch)

        val_loss = self.loss_fn(
            output_wave=predictions_wave, 
            output_wind=predictions_wind, 
            target_wind=target_wind, 
            target_wave=target_wave
        )

        log_dict = {"val_loss":val_loss}

        self.log_dict(log_dict, prog_bar=True)

        return log_dict


if __name__ == '__main__':
    print('python version: ', sys.version)
    print('optuna version: ', optuna.__version__)
    print('pytorch lightning version: ', pl.__version__)

    if version.parse(pl.__version__) < version.parse("1.6.0"):
        raise RuntimeError("PyTorch Lightning>=1.6.0 is required for this example.")

    pl.seed_everything(0, workers=True)
    
    sar_dir = '/mimer/NOBACKUP/priv/chair/sarssw/sar_dataset'
    fl_df_path = '/mimer/NOBACKUP/priv/chair/sarssw/sar_dataset_features_labels_27_april/sar_dataset_split.pickle'

    n_files = 10_000
    file_filter = lambda fn: fn.is_file() and fn.name.endswith('.nc') and 'IW' in fn.name
    sar_dataset_files = list(islice((fn.name for fn in os.scandir(sar_dir) if file_filter(fn)), n_files))
    #sar_dataset_files = [f for f in os.listdir(sar_dir) if f.endswith('.nc') and 'IW' in f]

    with open(fl_df_path, 'rb') as f:
        fl_df = pickle.load(f)

    fl_df = fl_df[fl_df.file_name.isin(sar_dataset_files)]

    def objective(trial: optuna.trial.Trial) -> float:
        
        # Suggest a learning rate
        learning_rate = learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        
        image_unit = trial.suggest_categorical('image_unit', ['lin', 'db'])
        
        train_dataset = CustomDataset(fl_df, sar_dir, split='train', scale_features=True, image_unit=image_unit)
        val_dataset = CustomDataset(fl_df, sar_dir, split='val', scale_features=True, image_unit=image_unit, mean=train_dataset.mean, std=train_dataset.std)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=16)
        
        feature_dim = train_dataset.feature_dim
        model = ImageFeatureRegressor(feature_dim=feature_dim, learning_rate=learning_rate, mean_wind=train_dataset.mean_wind, mean_wave=train_dataset.mean_wave)
        
        logger = pl.loggers.TensorBoardLogger("custom_loss", name=f"learning_rate={learning_rate}, image_unit={image_unit}")
        
        trainer = pl.Trainer(
            logger=logger,
            enable_checkpointing=True,
            max_epochs=50,
            accelerator="gpu",
            devices=1,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
            deterministic=True,
        )

        hyperparameters = dict(learning_rate=learning_rate, image_unit=image_unit)
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, train_loader, val_loader)
        return trainer.callback_metrics["val_loss"].item()
    
    # Create the Optuna study and optimize the objective function
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=10, timeout=3600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
