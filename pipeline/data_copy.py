#!/usr/bin/env python3
"""
This script is used to move the files for training, testing and/or validation to the tempdir on Alvis
"""

import pickle
import os
import sys
import argparse
import shutil

import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest',   required=True, help='The destination path on alvis, most likely the $TMPDIR variable in the slurm script')
    parser.add_argument('--src',    required=True, help='The path to the source folder of the images described in --data_df')
    parser.add_argument('--data_df', required=True, help='The path to the dataframe containg the metadata, features and labels for the images')
    parser.add_argument('--train', action='store_true', help='Flag to enable copying of the train data')
    parser.add_argument('--test', action='store_true', help='Flag to enable copying of the test data')
    parser.add_argument('--val', action='store_true', help='Flag to enable copying of the validation data')
    parser.add_argument('--debug', action='store_true', help='Used for debugging. Print the number of files that would be copied if this flag was not present')

    args = parser.parse_args()
    datasets = [(args.train, 'train'), (args.test, 'test'), (args.val, 'val')] 

    if not any([flag for (flag, _) in datasets]):
        print("No dataset specified to be copied")
        sys.exit(1)

    #Load features and labels dataframe 
    with open(args.data_df, 'rb') as f:
        fl_df = pickle.load(f)

    #filter data
    #Here we can add filtering to only keep certain types of data
    #fl_df = fl_df[fl_df['swath'] == 'IW']
    #fl_df = fl_df[fl_df['pol'].isin(['HH', 'HV'])]

    dataset_filter = [name for (flag, name) in datasets if flag]

    all_imgs = set(os.listdir(args.src))

    #only print debug information about the copy
    if args.debug:
        #Filter for the specified datasets
        print(f'dataset_filter:{dataset_filter}')
        fl_df = fl_df[fl_df['split'].isin(dataset_filter)]
        
        print(fl_df.columns)
        if fl_df.shape[0] > 0:
            print(f"filtered dataframe contains {fl_df.shape[0]} rows")
        copy_imgs = all_imgs.intersection(set(fl_df['file_name'].unique()))
        print(f"Without the --debug flag thesee parameters would result in {len(copy_imgs)} files being copied from {args.src} to {args.dest}")
    
    #Copy the actual data
    else:
        for dataset_name in dataset_filter:
            copy_dest = os.path.join(args.dest, dataset_name)
            #Create dictionary
            os.makedirs(copy_dest, exist_ok=True)

            fl_df_dataset = fl_df[fl_df['split'] == dataset_name]
            copy_imgs = all_imgs.intersection(set(fl_df_dataset['file_name'].unique()))
            print("Copying {dataset_name} dataset to {copy_dest}")
            #Copying the files
            for file in tqdm(copy_imgs): #TODO remove
                shutil.copyfile(os.path.join(args.src, file), os.path.join(copy_dest, file))