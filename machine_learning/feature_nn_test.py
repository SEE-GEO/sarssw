import argparse
import pandas as pd
import pickle
from tqdm import tqdm
import os

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
    parser.add_argument('--checkpoint', required=True, help='The path to a checkpoint to restart from.')
    parser.add_argument('--save_dir', required=True, help='The directory to save the predictions to.')
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

    train_dataset = sml.CustomDatasetFeatures(
        args.data_dir,
        args.dataframe_path,
        split='train',
        scale_features=True,
        get_names=False,
    )
    
    val_dataset = sml.CustomDatasetFeatures(
        args.data_dir,
        args.dataframe_path,
        split='val',
        scale_features=True, 
        feature_mean=model.feature_mean, 
        feature_std=model.feature_std, 
        get_names=False,
    )

    test_dataset = sml.CustomDatasetFeatures(
        args.data_dir, 
        args.dataframe_path, 
        split='test',
        scale_features=True, 
        feature_mean=model.feature_mean, 
        feature_std=model.feature_std, 
        get_names=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=128, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)

    train_result = trainer.validate(model, dataloaders=train_loader)
    train_result_str = "Validation on the train data:\n" + str(train_result)
    print(train_result_str)

    val_result = trainer.validate(model, dataloaders=val_loader)
    val_result_str = "Validation on the validation data:\n" + str(val_result)
    print(val_result_str)

    test_result = trainer.test(model, dataloaders=test_loader)
    test_result_str = "Test result on the test data:\n" + str(test_result)
    print(test_result_str)

    # Enable getting the file names from the dataset
    train_dataset.get_names = True
    val_dataset.get_names = True
    test_dataset.get_names = True

    model_name = args.checkpoint.split("/")[-1]
    save_dir = os.path.join(args.save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save validation and test results for the train, val and test data
    with open(os.path.join(save_dir, 'results.txt'), 'w') as f:
        f.write(train_result_str + '\n' + val_result_str + '\n' + test_result_str)

    for dataset_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        print(f"Predicting on {dataset_name} data")
        outputs = trainer.predict(model, dataloaders=loader)

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

        result_df = pd.DataFrame(data_dict)
        result_df.to_csv(os.path.join(save_dir, f"{dataset_name}_predictions.csv"))