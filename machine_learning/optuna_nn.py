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
        
        self.labels_cols = ['SWH_value_VV'] # TODO add back wspd
        labels_array = merge_df[self.labels_cols].values.astype(np.float32)
        self.labels_tensor = torch.from_numpy(labels_array)
        
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
        labels = self.labels_tensor[index]
        
        file_name = self.file_names.iloc[index]
        image_path = os.path.join(self.sar_dir, file_name)
        sigma0 = xr.open_dataset(image_path).sigma0.values.astype(np.float32)
        if self.image_unit == 'db':
            sigma0 = np.log10(np.where(sigma0 > 0.0, sigma0, 1e-10))
        image = torch.tensor(sigma0)

        return image, features, labels

class FeatureRegressor(pl.LightningModule):
    def __init__(self, feature_dim, fc_layers, learning_rate, optim_name='adam'):
        super(FeatureRegressor, self).__init__()
        self.learning_rate = learning_rate
        self.optim_name = optim_name
        
        self.image_cnn = resnet18(pretrained=False)
        self.image_cnn.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.image_cnn.fc = nn.Identity()

        # Define the fully connected layers
        fc_layers = [512 + feature_dim] + fc_layers + [1]
        self.fc_layers = nn.ModuleList()
        for i in range(len(fc_layers) - 1):
            self.fc_layers.append(nn.Linear(fc_layers[i], fc_layers[i + 1]))

    def forward(self, image, features):
        image_output = self.image_cnn(image)
        image_output = image_output.view(image_output.size(0), -1)
        
        combined = torch.cat((image_output, features), dim=1)
        
        # Pass the input through each fully connected layer
        x = combined
        for i, fc_layer in enumerate(self.fc_layers):
            x = F.relu(fc_layer(x)) if i < len(self.fc_layers) - 1 else fc_layer(x)
        
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
        if self.optim_name == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optim_name == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optim_name == 'rmsprop':
            optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optim_name}")

        return optimizer
    
    def validation_step(self, batch, batch_idx):
        image_batch, feature_batch, target_batch = batch
        predictions = self(image_batch, feature_batch)

        # Compute your evaluation metric(s)
        loss = nn.MSELoss()
        val_mse_loss = loss(predictions, target_batch)

        val_rmse = torch.sqrt(val_mse_loss)
        val_mae = torch.mean(torch.abs(predictions - target_batch))

        log_dict = {"val_mse_loss":val_mse_loss, "val_rmse":val_rmse, "val_mae":val_mae}

        self.log_dict(log_dict, prog_bar=True)

        # Return the metric(s) as a dictionary
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

    n_files = 100_000
    file_filter = lambda fn: fn.is_file() and fn.name.endswith('.nc') and 'IW' in fn.name
    sar_dataset_files = list(islice((fn.name for fn in os.scandir(sar_dir) if file_filter(fn)), n_files))
    #sar_dataset_files = [f for f in os.listdir(sar_dir) if f.endswith('.nc') and 'IW' in f]

    with open(fl_df_path, 'rb') as f:
        fl_df = pickle.load(f)

    fl_df = fl_df[fl_df.file_name.isin(sar_dataset_files)]

    def objective(trial: optuna.trial.Trial) -> float:
        # Suggest the number of layers and the size of each layer
        n_layers = trial.suggest_int('n_layers', 5, 10)
        fc_layers = [trial.suggest_int(f'n_units_l{i}', 4, 1024) for i in range(n_layers)]
        
        # Suggest a learning rate
        learning_rate = learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        
        # Suggest an optimizer name
        tune_optim = False 
        optim_name = 'adam' if not tune_optim else trial.suggest_categorical('optim_name', ['adam', 'sgd', 'rmsprop'])
        
        image_unit = trial.suggest_categorical('image_unit', ['lin', 'db'])
        
        train_dataset = CustomDataset(fl_df, sar_dir, split='train', scale_features=True, image_unit=image_unit)
        val_dataset = CustomDataset(fl_df, sar_dir, split='val', scale_features=True, image_unit=image_unit, mean=train_dataset.mean, std=train_dataset.std)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=16)
        
        feature_dim = train_dataset.feature_dim
        model = FeatureRegressor(feature_dim, fc_layers, learning_rate, optim_name)
        
        logger = pl.loggers.TensorBoardLogger("db_lin_sigma", name=f"learning_rate={learning_rate}, n_layers={n_layers}, fc_layers={fc_layers}, image_unit={image_unit}")
        
        trainer = pl.Trainer(
            logger=logger,
            enable_checkpointing=True,
            max_epochs=50,
            accelerator="gpu",
            devices=2,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_mse_loss")],
            deterministic=True,
        )

        hyperparameters = dict(learning_rate=learning_rate, n_layers=n_layers, fc_layers=fc_layers, image_unit=image_unit)
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, train_loader, val_loader)
        return trainer.callback_metrics["val_mse_loss"].item()
    
    # Create the Optuna study and optimize the objective function
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100, timeout=3600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
