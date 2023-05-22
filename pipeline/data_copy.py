#!/usr/bin/env python3
"""
This script is used to move the files for training, testing and/or validation to the tempdir on Alvis

For example running:
python data_copy.py --src /mimer/NOBACKUP/priv/chair/sarssw/sar_dataset/ --dest ${TMPDIR}/data --data_df /mimer/NOBACKUP/priv/chair/sarssw/sar_dataset_features_labels_27_april/sar_dataset_split.pickle --train --train_limit 1000 --val --val_limit 33 --non_nan --homogeneous --swaths 'IW' --polarisations 'VV VH'
In slurm would save 1000 train and 33 validation images in ${TMPDIR}/data filtered for non_nan values, homogeneouity, swaths=IW and polarisations=VV VH
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
    parser.add_argument('--dest', required=True, help='The destination path on alvis, most likely the $TMPDIR variable in the slurm script')
    parser.add_argument('--src', required=True, help='The path to the source folder of the images described in --data_df')
    parser.add_argument('--data_df', required=True, help='The path to the dataframe containg the metadata, features and labels for the images')
    parser.add_argument('--train', action='store_true', help='Flag to enable copying of the train dataset')
    parser.add_argument('--test', action='store_true', help='Flag to enable copying of the test dataset')
    parser.add_argument('--val', action='store_true', help='Flag to enable copying of the validation dataset')
    parser.add_argument('--train_limit', type=int, default=None, help='Set a limit on the number of files saved for the train dataset. Not specifying gives all data')
    parser.add_argument('--test_limit', type=int, default=None, help='Set a limit on the number of files saved for the test dataset. Not specifying gives all data')
    parser.add_argument('--val_limit', type=int, default=None, help='Set a limit on the number of files saved for the val dataset. Not specifying gives all data')
    parser.add_argument('--non_nan', action='store_true', help='Enables filtering for only data without nans')
    parser.add_argument('--homogeneous', action='store_true', help='Enables filtering for only data that passed the homogeneous filter')
    parser.add_argument('--swaths', help="Enables filtering to only include the specified swaths, seperate the swatchs with comma. i.e 'IW,EW' to include both IW and EW")
    parser.add_argument('--polarisations', help="Enables filtering to only include images with the specified polarisations, seperate the polarisations with comma. i.e 'VV VH,HH HV' would include files with both VV VH and HH HV polarisations")
    parser.add_argument('--debug', action='store_true', help='Used for debugging. Print the number of files that would be copied if this flag was not present')
    parser.add_argument('--file_name_filter', help="Regex filter to be used when filtering the file names. i.e 'BO|IR' would only include files with BO or IR in the file name")

    args = parser.parse_args()
    datasets = [(args.train, 'train', args.train_limit), (args.test, 'test', args.test_limit), (args.val, 'val', args.val_limit)] 

    if not any([flag for (flag, _, _) in datasets]):
        print("No dataset specified to be copied")
        sys.exit(1)

    #Load features and labels dataframe 
    with open(args.data_df, 'rb') as f:
        fl_df = pickle.load(f)

    #Filter for files without images with nan values
    if args.non_nan:
        fl_df_any_nan = fl_df
        fl_df_any_nan['any_nan_row_wise'] = fl_df_any_nan.isna().apply(any, axis=1)
        grouped_any_nan = fl_df_any_nan.groupby(['file_name', 'polarisations'])['any_nan_row_wise']
        fl_df = fl_df[~grouped_any_nan.transform("any")]

    #Filter for files with only homogeneous images
    if args.homogeneous:
        grouped_hom_test = fl_df.groupby(['file_name', 'polarisations'])['hom_test']
        fl_df = fl_df[grouped_hom_test.transform("all")]

    #Filter based on swaths
    if args.swaths is not None:
        swaths_filter = [s.strip() for s in args.swaths.split(',')]
        print(f"swaths_filter: {swaths_filter}")
        fl_df = fl_df[fl_df['swath'].isin(swaths_filter)]

    #Filter based on polarisations
    if args.polarisations is not None:
        pol_filter = [p.strip() for p in args.polarisations.split(',')]
        print(f"pol_filter: {pol_filter}")
        fl_df = fl_df[fl_df['polarisations'].isin(pol_filter)]

    #filter based on file name
    if args.file_name_filter is not None:
        print(f"file_name_filter: {args.file_name_filter}")
        fl_df = fl_df[fl_df['file_name'].str.contains(args.file_name_filter)]

    dataset_filter = [(name, limit) for (flag, name, limit) in datasets if flag]

    all_imgs = set(os.listdir(args.src))

    #only print debug information about the copy
    if args.debug:
        print(f"Without the --debug flag this command would result in:")

    for dataset_name, dataset_limit in dataset_filter:
        copy_dest = os.path.join(args.dest, dataset_name)
        #Create dictionary
        os.makedirs(copy_dest, exist_ok=True)

        fl_df_dataset = fl_df[fl_df['split'] == dataset_name]
        copy_imgs = all_imgs.intersection(set(fl_df_dataset['file_name'].unique()[:dataset_limit]))

        #only print debug information about the copy
        if args.debug:
            print(f"dataset {dataset_name}: {len(copy_imgs)} files being copied to {copy_dest}")
    
        #Copy the actual data
        else:
            print(f"Copying {dataset_name} dataset to {copy_dest}")
            #Copying the files
            for file in tqdm(copy_imgs):
                shutil.copyfile(os.path.join(args.src, file), os.path.join(copy_dest, file))