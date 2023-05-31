from packaging import version
import sys
import argparse
import pandas as pd
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
import pytorch_lightning as pl
import optuna

import sarssw_ml_lib as sml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True, help='The path to the data on alvis, most likely something similar to $TMPDIR/data in the slurm script')
    parser.add_argument('--dataframe_path', required=True, help='The path to the dataframe containing the metadata, features and labels for the data.')
    parser.add_argument('--gpus', type=int, required=True, help='Specify the number of GPUs to use for training. They should be requested in the slurm scrips')
    parser.add_argument('--checkpoint', required=True, help='Optional. The path to a checkpoint to restart from.')
    args = parser.parse_args()

    print('python version: ', sys.version)
    print('optuna version: ', optuna.__version__)
    print('pytorch lightning version: ', pl.__version__)

    if version.parse(pl.__version__) < version.parse("1.6.0"):
        raise RuntimeError("PyTorch Lightning>=1.6.0 is required for this example.")

    pl.seed_everything(0, workers=True)

    #Load saved model from checkpoint
    model = sml.ImageFeatureRegressor.load_from_checkpoint(args.checkpoint)

    # disable randomness, dropout, etc...
    model.eval()

    normalize_transform = Normalize(mean=model.pixel_mean, std=model.pixel_std)
    
    val_dataset = sml.CustomDataset(
        args.data_dir, 
        args.dataframe_path, 
        split='val',
        scale_features=True, 
        feature_mean=model.feature_mean, 
        feature_std=model.feature_std, 
        transform=normalize_transform
    )

    test_dataset = sml.CustomDataset(
        args.data_dir, 
        args.dataframe_path, 
        split='test',
        scale_features=True, 
        feature_mean=model.feature_mean, 
        feature_std=model.feature_std, 
        transform=normalize_transform
    )

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
    )

    test_result = trainer.test(model, test_dataloaders=test_loader)
    print(test_result)

    val_result = trainer.validate(model, dataloaders=val_loader)
    print(val_result)

    test_result_df_columns = ['target_wave', 'target_wind', 'predictions_wave', 'predictions_wind']
    test_result_df = pd.DataFrame({c: pd.Series(dtype=float) for c in test_result_df_columns})


    for batch in val_loader:
        image_batch, feature_batch, (target_wave, target_wind) = batch
        predictions_wave, predictions_wind = model(image_batch, feature_batch)
        predictions_wave = predictions_wave.squeeze(-1)  # remove the extra dimension
        predictions_wind = predictions_wind.squeeze(-1)  # remove the extra dimension

        stacked_tensor = pd.DataFrame(torch.stack([t.detach() for t in [target_wave, target_wind, predictions_wave, predictions_wind]], dim=1), columns=test_result_df_columns)

        test_result_df = pd.concat([test_result_df, stacked_tensor], ignore_index=True)
    
    print(test_result_df)
    print(test_result_df.shape)
    #with open('/mimer/NOBACKUP/priv/chair/sarssw/pickle_df/test_result_df.pickle', 'wb') as f:
    #    pickle.dump(test_result_df, f)
