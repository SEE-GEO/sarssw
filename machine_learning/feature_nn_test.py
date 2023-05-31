import argparse
import pandas as pd
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
import pytorch_lightning as pl

import sarssw_ml_lib as sml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True, help='The path to the data on alvis, most likely something similar to $TMPDIR/data in the slurm script')
    parser.add_argument('--dataframe_path', required=True, help='The path to the dataframe containing the metadata, features and labels for the data.')
    parser.add_argument('--gpus', type=int, required=True, help='Specify the number of GPUs to use for training. They should be requested in the slurm scrips')
    parser.add_argument('--checkpoint', required=True, help='Optional. The path to a checkpoint to restart from.')
    args = parser.parse_args()

    #pl.seed_everything(0, workers=True)

    #Load saved model from checkpoint
    model = sml.FeatureRegressor.load_from_checkpoint(args.checkpoint)

    # disable randomness, dropout, etc...
    model.eval()

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=args.gpus,
    )
    
    normalize_transform = Normalize(mean=model.pixel_mean, std=model.pixel_std)

    test_dataset = sml.CustomDatasetFeatures(
        args.data_dir, 
        args.dataframe_path, 
        split='test',
        scale_features=True, 
        mean=model.feature_mean, 
        std=model.feature_std, 
    )

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)

    test_result = trainer.test(model, dataloaders=test_loader)

    outputs = trainer.predict(model, dataloaders=test_loader)

    file_names = [item for sublist in outputs for item in sublist['file_name']]
    target_wave = torch.cat([x['target_wave'] for x in outputs]).detach().cpu().numpy()
    target_wind = torch.cat([x['target_wind'] for x in outputs]).detach().cpu().numpy()
    prediction_wave = torch.cat([x['prediction_wave'] for x in outputs]).detach().cpu().numpy()
    prediction_wind = torch.cat([x['prediction_wind'] for x in outputs]).detach().cpu().numpy()

    data_dict = {
        "file_name": file_names,
        "target_wave": target_wave,
        "target_wind": target_wind,
        "prediction_wave": prediction_wave,
        "prediction_wind": prediction_wind
    }

    df_test = pd.DataFrame(data_dict)
    df_test.to_csv("/cephyr/users/brobeck/Alvis/sarssw/sandbox/test_results_feat.csv")