#!/usr/bin/env python3
"""
This script is used to convert already existing netcdf files to 32 bit float
"""

import pickle
import os
import sys
import argparse
import shutil

import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', required=True, help='The path to the source folder of the netcdf files')
    parser.add_argument('--dest', required=True, help='The destination path where the converted files will be saved. This must be empty')
    args = parser.parse_args()
    
    if not os.path.isdir(args.src):
        raise ValueError(f"src: {args.src} is not a directory")
    
    if os.path.isdir(args.dest):
        if os.listdir(args.dest):
            raise ValueError(f"dest: {args.dest} is not empty")
    else:
        #Create the destination folder if it does not exist
        os.makedirs(args.dest, exist_ok=True)
    
    #Get the list of files to convert
    files = os.listdir(args.src)
       
    
    
    #Convert the files
    for file in tqdm(files):
        xds = xr.open_dataset(os.path.join(args.src, file))
        xds32 = xds.astype(np.float32)
        xds32.to_netcdf(path=os.path.join(args.dest, file))

