import pickle
import os
import random

import numpy as np
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

# In large parts taken from https://kozodoi.me/blog/20210308/compute-image-stats
def dataset_mean_std(dataset):
    "Returns the pixel mean and standared deviation of a dataset"
    dataset_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
    
    n_channels, x_pixels, y_pixels = dataset[0][0].shape
    pixel_value_sum = torch.tensor([0.0]*n_channels)
    pixel_value_squared_sum = torch.tensor([0.0]*n_channels)

    # loop through images
    print("Computing mean and std")
    for images, features, labels in tqdm(dataset_loader):
        pixel_value_sum += images.sum(axis = [0, 2, 3])
        pixel_value_squared_sum += (images ** 2).sum(axis = [0, 2, 3])

    # pixel count
    pixel_count = len(dataset) * x_pixels * y_pixels

    # mean and std
    pixel_mean = pixel_value_sum / pixel_count
    pixel_var  = (pixel_value_squared_sum / pixel_count) - (pixel_mean ** 2)
    pixel_std  = torch.sqrt(pixel_var)

    return pixel_mean, pixel_std

class RandomRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return TF.rotate(img, angle)

class CustomDataset(Dataset):
    def __init__(self, data_dir, dataframe_path, split='train', base_features=None, scale_features=True, feature_mean=None, feature_std=None, transform=None):
        self.split_dir = os.path.join(data_dir, split)
        
        if not os.path.isdir(self.split_dir):
            raise ValueError(f"The data directory {self.split_dir} not found")

        with open(dataframe_path, 'rb') as f:
            fl_df = pickle.load(f)

        #Filter for only images in the data_dir
        dataset_df = fl_df[fl_df.file_name.isin(os.listdir(self.split_dir))]
        
        # preprocess with hom test, feature extract and everything
        #filter homogenious images with IW mode and no na
        split_df = dataset_df[dataset_df.split == split]
        hom_df = split_df[split_df.hom_test].copy()
        
        hom_df = hom_df.dropna()
        #merge the vv and vh polarization
        VV_df, VH_df = hom_df[hom_df.pol == 'VV'], hom_df[hom_df.pol == 'VH']
        merge_df = VV_df.merge(VH_df, on='file_name', suffixes=('_VV', '_VH'))

        if base_features is None:
            base_features = [
                'contrast', 'dissimilarity', 'homogeneity', 
                'energy', 'correlation', 'ASM', 
                'sigma_mean', 'sigma_var', 'sigma_mean_over_var', 
                'sigma_min', 'sigma_max', 'sigma_range',
                'acw', 'acw_median', 'acw_db', 'acw_median_db'
            ]
        
        self.features = [f + p for f in base_features for p in ['_VV', '_VH']] + ['incidence_VV']
        self.feature_dim = len(self.features) 
        features_array = merge_df[self.features].values.astype(np.float32)
        features_tensor = torch.from_numpy(features_array)

        if scale_features:
            # If the dataset is the training set, fit and transform the features
            if split == 'train':
                self.feature_mean = features_tensor.mean(dim=0)
                self.feature_std = features_tensor.std(dim=0)
                features_tensor = (features_tensor - self.feature_mean) / self.feature_std
            # If the dataset is the validation or test set, transform the features using the mean and std computed on the training set
            elif feature_mean is not None and feature_std is not None:
                self.feature_mean = feature_mean
                self.feature_std = feature_std
                features_tensor = (features_tensor - self.feature_mean) / self.feature_std

        self.features_tensor = features_tensor      
        
        self.wave_col = 'SWH_value_VV'
        self.mean_wave = merge_df[self.wave_col].mean()
        self.wave_tensor = torch.from_numpy(merge_df[self.wave_col].values.astype(np.float32))
        self.wave_source = merge_df['SWH_source_VV']
        
        self.wind_col = 'WSPD_value_VV'
        self.mean_wind = merge_df[self.wind_col].mean()
        self.wind_tensor = torch.from_numpy(merge_df[self.wind_col].values.astype(np.float32))
        self.wind_source = merge_df['WSPD_source_VV']

        self.split = split
        self.file_names = merge_df.file_name
        
        self.n_wave_bouy = (self.wave_source == 'bouy').sum()            
        self.n_wind_bouy = (self.wind_source == 'bouy').sum()
        self.transform = transform

    def __len__(self):
        return len(self.features_tensor)

    def __getitem__(self, index):
        features = self.features_tensor[index]
        wave_label, wind_label = self.wave_tensor[index], self.wind_tensor[index]
        
        file_name = self.file_names.iloc[index]
        image_path = os.path.join(self.split_dir, file_name)
        sigma0 = xr.open_dataset(image_path).sigma0.values
        image = torch.tensor(sigma0)
        
        # Apply transform if it exists
        if self.transform is not None:
            image = self.transform(image)  

        if self.split != 'train':
            # If not training set return -1 for labels that are from model
            if self.wave_source[index] != 'bouy':
                wave_label = torch.tensor(-1.0)
            if self.wind_source[index] != 'bouy':
                wind_label = torch.tensor(-1.0)
        if self.split != 'test':        
            return image, features, (wave_label, wind_label)
        return image, features, (wave_label, wind_label), file_name
    
class CustomDatasetFeatures(Dataset):
    def __init__(self, data_dir, dataframe_path, split='train', base_features=None, scale_features=True, feature_mean=None, feature_std=None, transform=None):
        self.split_dir = os.path.join(data_dir, split)
        
        if not os.path.isdir(self.split_dir):
            raise ValueError(f"The data directory {self.split_dir} not found")

        with open(dataframe_path, 'rb') as f:
            fl_df = pickle.load(f)

        #Filter for only images in the data_dir
        dataset_df = fl_df[fl_df.file_name.isin(os.listdir(self.split_dir))]
        
        # preprocess with hom test, feature extract and everything
        #filter homogenious images with IW mode and no na
        split_df = dataset_df[dataset_df.split == split]
        hom_df = split_df[split_df.hom_test].copy()
        
        hom_df = hom_df.dropna()
        #merge the vv and vh polarization
        VV_df, VH_df = hom_df[hom_df.pol == 'VV'], hom_df[hom_df.pol == 'VH']
        merge_df = VV_df.merge(VH_df, on='file_name', suffixes=('_VV', '_VH'))

        if base_features is None:
            base_features = [
                'contrast', 'dissimilarity', 'homogeneity', 
                'energy', 'correlation', 'ASM', 
                'sigma_mean', 'sigma_var', 'sigma_mean_over_var', 
                'sigma_min', 'sigma_max', 'sigma_range',
                'acw', 'acw_median', 'acw_db', 'acw_median_db'
            ]
        
        self.features = [f + p for f in base_features for p in ['_VV', '_VH']] + ['incidence_VV']
        self.feature_dim = len(self.features) 
        features_array = merge_df[self.features].values.astype(np.float32)
        features_tensor = torch.from_numpy(features_array)

        if scale_features:
            # If the dataset is the training set, fit and transform the features
            if split == 'train':
                self.feature_mean = features_tensor.mean(dim=0)
                self.feature_std = features_tensor.std(dim=0)
                features_tensor = (features_tensor - self.feature_mean) / self.feature_std
            # If the dataset is the validation or test set, transform the features using the mean and std computed on the training set
            elif feature_mean is not None and feature_std is not None:
                self.feature_mean = feature_mean
                self.feature_std = feature_std
                features_tensor = (features_tensor - self.feature_mean) / self.feature_std

        self.features_tensor = features_tensor      
        
        self.wave_col = 'SWH_value_VV'
        self.mean_wave = merge_df[self.wave_col].mean()
        self.wave_tensor = torch.from_numpy(merge_df[self.wave_col].values.astype(np.float32))
        self.wave_source = merge_df['SWH_source_VV']
        
        self.wind_col = 'WSPD_value_VV'
        self.mean_wind = merge_df[self.wind_col].mean()
        self.wind_tensor = torch.from_numpy(merge_df[self.wind_col].values.astype(np.float32))
        self.wind_source = merge_df['WSPD_source_VV']

        self.split = split
        self.file_names = merge_df.file_name
        
        self.n_wave_bouy = (self.wave_source == 'bouy').sum()            
        self.n_wind_bouy = (self.wind_source == 'bouy').sum()

    def __len__(self):
        return len(self.features_tensor)

    def __getitem__(self, index):
        features = self.features_tensor[index]
        wave_label, wind_label = self.wave_tensor[index], self.wind_tensor[index]

        if self.split != 'train':
            # If not training set return -1 for labels that are from model
            if self.wave_source[index] != 'bouy':
                wave_label = torch.tensor(-1.0)
            if self.wind_source[index] != 'bouy':
                wind_label = torch.tensor(-1.0)
                
        if self.split != 'test':        
            return features, (wave_label, wind_label)
        return features, (wave_label, wind_label), self.file_names.iloc[index]

class CustomLoss(nn.Module):
    def __init__(self, mean_wave, mean_wind):
        super(CustomLoss, self).__init__()
        self.mean_wave = mean_wave
        self.mean_wind = mean_wind

    def forward(self, output_wave, output_wind, target_wave, target_wind):   
        mask_wave = target_wave != -1
        mask_wind = target_wind != -1

        mse_loss = nn.MSELoss(reduction='sum')  # use sum to ignore the masked entries

        # calculate metrics only for valid entries
        n_wave_bouy = mask_wave.sum()
        wave_mse = mse_loss(output_wave[mask_wave] / self.mean_wave, target_wave[mask_wave] / self.mean_wave) / n_wave_bouy
        #wave_rmse = torch.sqrt(wave_mse)

        n_wind_bouy = mask_wind.sum()
        wind_mse = mse_loss(output_wind[mask_wind] / self.mean_wind, target_wind[mask_wind] / self.mean_wind) / n_wind_bouy
        #wind_rmse = torch.sqrt(wind_mse)
        
        loss = torch.sqrt((wave_mse + wind_mse) / 2)
        return loss

class ImageFeatureRegressor(pl.LightningModule):
    def __init__(self, model_name, pretrained, feature_dim, learning_rate, mean_wind, mean_wave, dropout_p=0.5, feature_mean=None, feature_std=None, pixel_mean=None, pixel_std=None):
        super(ImageFeatureRegressor, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.loss_fn = CustomLoss(mean_wave=mean_wave, mean_wind=mean_wind)
        self.dropout_p = dropout_p

        self.feature_mean=feature_mean
        self.feature_std=feature_std

        self.pixel_mean=pixel_mean
        self.pixel_std=pixel_std
        
        #self.image_cnn = resnet18()  # Default to no pre-training
        #self.image_cnn.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.image_cnn.fc = nn.Identity()

        # Use the given CNN model name to load the suggested pretrained model
        if model_name == 'resnet18':
            self.image_cnn = resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            self.image_cnn = resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            self.image_cnn = resnet50(pretrained=pretrained)
        elif model_name == 'vgg16':
            self.image_cnn = vgg16(pretrained=pretrained)
        elif model_name == 'vgg19':
            self.image_cnn = vgg19(pretrained=pretrained)


        num_cnn_out_features = 0
        if isinstance(self.image_cnn, resnet.ResNet):
            self.image_cnn.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_cnn_out_features = self.image_cnn.fc.in_features  # Get the number of output features before setting fc to Identity
            self.image_cnn.fc = nn.Identity()  # Ensure the model does not include the final fully connected layer
        elif isinstance(self.image_cnn, vgg.VGG):
            self.image_cnn.features[0] = nn.Conv2d(2, 64, kernel_size=3, padding=1)
            num_cnn_out_features = self.image_cnn.classifier[0].in_features  # VGG's fully connected layer is in the classifier attribute
            self.image_cnn.classifier = nn.Identity()

        self.fc1 = nn.Linear(num_cnn_out_features + feature_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.bn5 = nn.BatchNorm1d(1024)

        fc_sizes_wave = [1024, 512, 256, 128, 64]
        self.fc_wave = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(fc_sizes_wave[:-1], fc_sizes_wave[1:])])
        self.bn_wave = nn.ModuleList([nn.BatchNorm1d(size) for size in fc_sizes_wave[1:]])
        self.fc6_wave = nn.Linear(64, 1)

        fc_sizes_wind = [1024, 512, 256, 128, 64]
        self.fc_wind = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(fc_sizes_wind[:-1], fc_sizes_wind[1:])])
        self.bn_wind = nn.ModuleList([nn.BatchNorm1d(size) for size in fc_sizes_wind[1:]])
        self.fc6_wind = nn.Linear(64, 1)
        
    def forward(self, image, features):
        #image_output = self.image_cnn(image)
        #image_output = image_output.view(image_output.size(0), -1)
        
        image_output = self.image_cnn(image)

        if isinstance(self.image_cnn, resnet.ResNet):
            # If the CNN model is a ResNet, flatten the output tensor
            image_output = image_output.view(image_output.size(0), -1)
        # else: If the CNN model is a VGG, the output tensor is already flattened
        
        combined = torch.cat((image_output, features), dim=1)
        
        x = F.relu(self.bn1(self.fc1(combined)))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        # branch wave
        x_wave = x
        for fc, bn in zip(self.fc_wave, self.bn_wave):
            x_wave = F.relu(bn(fc(x_wave)))
            x_wave = F.dropout(x_wave, p=self.dropout_p, training=self.training)
        x_wave = self.fc6_wave(x_wave)
        
        # branch wind
        x_wind = x
        for fc, bn in zip(self.fc_wind, self.bn_wind):
            x_wind = F.relu(bn(fc(x_wind)))
            x_wind = F.dropout(x_wind, p=self.dropout_p, training=self.training)
        x_wind = self.fc6_wind(x_wind)
        
        return x_wave, x_wind

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
            
    def training_step(self, batch, batch_idx):
        image_batch, feature_batch, (target_wave, target_wind) = batch
        output_wave, output_wind = self(image_batch, feature_batch)
        
        output_wave = output_wave.squeeze(-1)  # remove the extra dimension
        output_wind = output_wind.squeeze(-1)  # remove the extra dimension
        
        loss = self.loss_fn(
            output_wave=output_wave, 
            output_wind=output_wind, 
            target_wave=target_wave,
            target_wind=target_wind, 
        )
        
        # Ignore everything within this context in the back prop
        with torch.no_grad():
            mse_loss = nn.MSELoss()
            wave_mse = mse_loss(output_wave, target_wave)
            wave_rmse = torch.sqrt(wave_mse)
            wave_mae = torch.mean(torch.abs(output_wave - target_wave))

            wind_mse = mse_loss(output_wind, target_wind)
            wind_rmse = torch.sqrt(wind_mse)
            wind_mae = torch.mean(torch.abs(output_wind - target_wind))

        log_dict = {
            "loss": loss,
            "train_wave_rmse": wave_rmse, 
            "train_wind_rmse": wind_rmse, 
            "train_wave_mae": wave_mae,
            "train_wind_mae": wind_mae,
        }
        
        # Log only selected metrics for the progress bar
        self.log_dict(log_dict, prog_bar=True)

        return log_dict
    
    def validation_step(self, batch, batch_idx):
        image_batch, feature_batch, (target_wave, target_wind) = batch
        predictions_wave, predictions_wind = self(image_batch, feature_batch)
        predictions_wave = predictions_wave.squeeze(-1)  # remove the extra dimension
        predictions_wind = predictions_wind.squeeze(-1)  # remove the extra dimension
        
        with torch.no_grad():
            # calculate loss
            val_loss = self.loss_fn(
                output_wave=predictions_wave, 
                output_wind=predictions_wind, 
                target_wave=target_wave,
                target_wind=target_wind, 
            )
            
            # create masks where target values are not -1
            mask_wave = target_wave != -1
            mask_wind = target_wind != -1

            mse_loss = nn.MSELoss(reduction='sum')  # use sum to ignore the masked entries

            # calculate metrics only for valid entries
            n_wave_bouy = mask_wave.sum()
            wave_mse = mse_loss(predictions_wave[mask_wave], target_wave[mask_wave]) / n_wave_bouy
            wave_rmse = torch.sqrt(wave_mse)
            wave_mae = torch.mean(torch.abs(predictions_wave[mask_wave] - target_wave[mask_wave]))

            n_wind_bouy = mask_wind.sum()
            wind_mse = mse_loss(predictions_wind[mask_wind], target_wind[mask_wind]) / n_wind_bouy
            wind_rmse = torch.sqrt(wind_mse)
            wind_mae = torch.mean(torch.abs(predictions_wind[mask_wind] - target_wind[mask_wind]))

        log_dict = {
            "val_loss": val_loss,
            "val_wave_rmse": wave_rmse, 
            "val_wind_rmse": wind_rmse, 
            "val_wave_mae": wave_mae,
            "val_wind_mae": wind_mae,
        }
        
        # Log only selected metrics for the progress bar
        self.log_dict(log_dict, prog_bar=True)

        return log_dict
    
    def test_step(self, batch, batch_idx):        
        image_batch, feature_batch, (target_wave, target_wind), _ = batch
        predictions_wave, predictions_wind = self(image_batch, feature_batch)
        predictions_wave = predictions_wave.squeeze(-1)  # remove the extra dimension
        predictions_wind = predictions_wind.squeeze(-1)  # remove the extra dimension
        
        with torch.no_grad():
            # calculate loss
            val_loss = self.loss_fn(
                output_wave=predictions_wave, 
                output_wind=predictions_wind, 
                target_wave=target_wave,
                target_wind=target_wind, 
            )
            
            # create masks where target values are not -1
            mask_wave = target_wave != -1
            mask_wind = target_wind != -1

            mse_loss = nn.MSELoss(reduction='sum')  # use sum to ignore the masked entries

            # calculate metrics only for valid entries
            n_wave_bouy = mask_wave.sum()
            wave_mse = mse_loss(predictions_wave[mask_wave], target_wave[mask_wave]) / n_wave_bouy
            wave_rmse = torch.sqrt(wave_mse)
            wave_mae = torch.mean(torch.abs(predictions_wave[mask_wave] - target_wave[mask_wave]))

            n_wind_bouy = mask_wind.sum()
            wind_mse = mse_loss(predictions_wind[mask_wind], target_wind[mask_wind]) / n_wind_bouy
            wind_rmse = torch.sqrt(wind_mse)
            wind_mae = torch.mean(torch.abs(predictions_wind[mask_wind] - target_wind[mask_wind]))

        log_dict = {
            "test_loss": val_loss,
            "test_wave_rmse": wave_rmse, 
            "test_wind_rmse": wind_rmse, 
            "test_wave_mae": wave_mae,
            "test_wind_mae": wind_mae,
        }
        
        self.log_dict(log_dict, prog_bar=True)

        return log_dict
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        image_batch, feature_batch, (target_wave, target_wind), file_names = batch
        predictions_wave, predictions_wind = self(image_batch, feature_batch)
        predictions_wave = predictions_wave.squeeze(-1)
        predictions_wind = predictions_wind.squeeze(-1)

        return {"file_name": file_names,
                "target_wave": target_wave,
                "target_wind": target_wind,
                "prediction_wave": predictions_wave,
                "prediction_wind": predictions_wind}
    
    
class FeatureRegressor(pl.LightningModule):
    def __init__(self, feature_dim, learning_rate, mean_wind, mean_wave, dropout_p=0.5, feature_mean=None, feature_std=None):
        super(FeatureRegressor, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.loss_fn = CustomLoss(mean_wave=mean_wave, mean_wind=mean_wind)
        self.dropout_p = dropout_p
        
        self.feature_mean = feature_mean      
        self.feature_std = feature_std 
        
        self.fc1 = nn.Linear(feature_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.bn5 = nn.BatchNorm1d(1024)

        fc_sizes_wave = [1024, 512, 256, 128, 64]
        self.fc_wave = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(fc_sizes_wave[:-1], fc_sizes_wave[1:])])
        self.bn_wave = nn.ModuleList([nn.BatchNorm1d(size) for size in fc_sizes_wave[1:]])
        self.fc6_wave = nn.Linear(64, 1)

        fc_sizes_wind = [1024, 512, 256, 128, 64]
        self.fc_wind = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(fc_sizes_wind[:-1], fc_sizes_wind[1:])])
        self.bn_wind = nn.ModuleList([nn.BatchNorm1d(size) for size in fc_sizes_wind[1:]])
        self.fc6_wind = nn.Linear(64, 1)
        
    def forward(self, features):
        x = F.relu(self.bn1(self.fc1(features)))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        # branch wave
        x_wave = x
        for fc, bn in zip(self.fc_wave, self.bn_wave):
            x_wave = F.relu(bn(fc(x_wave)))
            x_wave = F.dropout(x_wave, p=self.dropout_p, training=self.training)
        x_wave = self.fc6_wave(x_wave)
        
        # branch wind
        x_wind = x
        for fc, bn in zip(self.fc_wind, self.bn_wind):
            x_wind = F.relu(bn(fc(x_wind)))
            x_wind = F.dropout(x_wind, p=self.dropout_p, training=self.training)
        x_wind = self.fc6_wind(x_wind)
        
        return x_wave, x_wind

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
            
    def training_step(self, batch, batch_idx):
        feature_batch, (target_wave, target_wind) = batch
        output_wave, output_wind = self(feature_batch)
        
        output_wave = output_wave.squeeze(-1)  # remove the extra dimension
        output_wind = output_wind.squeeze(-1)  # remove the extra dimension
    
        loss = self.loss_fn(
            output_wave=output_wave, 
            output_wind=output_wind, 
            target_wave=target_wave,
            target_wind=target_wind, 
        )
            
        # Ignore everything within this context in the back prop
        with torch.no_grad():
            mse_loss = nn.MSELoss()
            wave_mse = mse_loss(output_wave, target_wave)
            wave_rmse = torch.sqrt(wave_mse)
            wave_mae = torch.mean(torch.abs(output_wave - target_wave))

            wind_mse = mse_loss(output_wind, target_wind)
            wind_rmse = torch.sqrt(wind_mse)
            wind_mae = torch.mean(torch.abs(output_wind - target_wind))

        log_dict = {
            "loss": loss,
            "train_wave_rmse": wave_rmse, 
            "train_wind_rmse": wind_rmse, 
            "train_wave_mae": wave_mae,
            "train_wind_mae": wind_mae,
        }
        
        # Log only selected metrics for the progress bar
        self.log_dict(log_dict, prog_bar=True)

        return log_dict
    
    def validation_step(self, batch, batch_idx):
        feature_batch, (target_wave, target_wind) = batch
        predictions_wave, predictions_wind = self(feature_batch)
        predictions_wave = predictions_wave.squeeze(-1)  # remove the extra dimension
        predictions_wind = predictions_wind.squeeze(-1)  # remove the extra dimension

        with torch.no_grad():
            # calculate loss
            val_loss = self.loss_fn(
                output_wave=predictions_wave, 
                output_wind=predictions_wind, 
                target_wave=target_wave,
                target_wind=target_wind, 
            )

            # create masks where target values are not -1
            mask_wave = target_wave != -1
            mask_wind = target_wind != -1

            mse_loss = nn.MSELoss(reduction='sum')  # use sum to ignore the masked entries

            # calculate metrics only for valid entries
            n_wave_bouy = mask_wave.sum()
            wave_mse = mse_loss(predictions_wave[mask_wave], target_wave[mask_wave]) / n_wave_bouy
            wave_rmse = torch.sqrt(wave_mse)
            wave_mae = torch.mean(torch.abs(predictions_wave[mask_wave] - target_wave[mask_wave]))

            n_wind_bouy = mask_wind.sum()
            wind_mse = mse_loss(predictions_wind[mask_wind], target_wind[mask_wind]) / n_wind_bouy
            wind_rmse = torch.sqrt(wind_mse)
            wind_mae = torch.mean(torch.abs(predictions_wind[mask_wind] - target_wind[mask_wind]))

        log_dict = {
            "val_loss": val_loss,
            "val_wave_rmse": wave_rmse, 
            "val_wind_rmse": wind_rmse, 
            "val_wave_mae": wave_mae,
            "val_wind_mae": wind_mae,
        }
        
        # Log only selected metrics for the progress bar
        self.log_dict(log_dict, prog_bar=True)

        return log_dict
    
    def test_step(self, batch, batch_idx):        
        image_batch, feature_batch, (target_wave, target_wind), _ = batch
        predictions_wave, predictions_wind = self(image_batch, feature_batch)
        predictions_wave = predictions_wave.squeeze(-1)  # remove the extra dimension
        predictions_wind = predictions_wind.squeeze(-1)  # remove the extra dimension
        
        with torch.no_grad():
            # calculate loss
            val_loss = self.loss_fn(
                output_wave=predictions_wave, 
                output_wind=predictions_wind, 
                target_wave=target_wave,
                target_wind=target_wind, 
            )
            
            # create masks where target values are not -1
            mask_wave = target_wave != -1
            mask_wind = target_wind != -1

            mse_loss = nn.MSELoss(reduction='sum')  # use sum to ignore the masked entries

            # calculate metrics only for valid entries
            n_wave_bouy = mask_wave.sum()
            wave_mse = mse_loss(predictions_wave[mask_wave], target_wave[mask_wave]) / n_wave_bouy
            wave_rmse = torch.sqrt(wave_mse)
            wave_mae = torch.mean(torch.abs(predictions_wave[mask_wave] - target_wave[mask_wave]))

            n_wind_bouy = mask_wind.sum()
            wind_mse = mse_loss(predictions_wind[mask_wind], target_wind[mask_wind]) / n_wind_bouy
            wind_rmse = torch.sqrt(wind_mse)
            wind_mae = torch.mean(torch.abs(predictions_wind[mask_wind] - target_wind[mask_wind]))

        log_dict = {
            "test_loss": val_loss,
            "test_wave_rmse": wave_rmse, 
            "test_wind_rmse": wind_rmse, 
            "test_wave_mae": wave_mae,
            "test_wind_mae": wind_mae,
        }
        
        self.log_dict(log_dict, prog_bar=True)

        return log_dict
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        feature_batch, (target_wave, target_wind), file_names = batch
        predictions_wave, predictions_wind = self(feature_batch)
        predictions_wave = predictions_wave.squeeze(-1)
        predictions_wind = predictions_wind.squeeze(-1)

        return {"file_name": file_names,
                "target_wave": target_wave,
                "target_wind": target_wind,
                "prediction_wave": predictions_wave,
                "prediction_wind": predictions_wind}
    