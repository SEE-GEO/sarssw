from packaging import version
import sys
import argparse

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

    print(args.checkpoint) #TODO remove
  
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
    
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
    
    trainer = pl.Trainer()
    print(trainer.validate(model, dataloaders=val_loader))
  