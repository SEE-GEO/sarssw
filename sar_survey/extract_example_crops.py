import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from collections import defaultdict

import pickle
from tqdm import tqdm
from zipfile import ZipFile
import os
from time import sleep

import itertools

import xsar
import xarray as xr


download_dir = '/data/exjobb/sarssw/sar/'
out_dir = '/data/exjobb/sarssw/sar_segments/'
print('Extracting sar images around bouys')
print('From:', download_dir, 'To:', out_dir)

write_folder = '/home/sarssw/axel/sarssw/bouy_survey/1h_survey'
result_df_fn = 'result_df'

with open(os.path.join(write_folder, result_df_fn),'rb') as f_r:
    shore_survey_df = pickle.load(f_r)

record_dict = defaultdict(list)
name_counter = 0
box_size = 2000 # 2km

for file_name in tqdm(os.listdir(download_dir)):
     if file_name.endswith('.SAFE'):
        full_name = download_dir + file_name
        img_name = file_name.split('.')[0]
        
        # atrack = line, xtrack = sample
        sar_meta = xsar.Sentinel1Meta(full_name)
        sar_ds = xsar.Sentinel1Dataset(sar_meta)
        dist = {
            'atrack': int(np.round(box_size / 2 / sar_meta.pixel_atrack_m)),
            'xtrack': int(np.round(box_size / 2 / sar_meta.pixel_xtrack_m))
        }
        
        shore_df_part = shore_survey_df[
            shore_survey_df.sar_url.str.contains(img_name)
        ]
        
        for bouy, bouy_df in shore_df_part.groupby('bouy_file_name'):
            bouy_lon, bouy_lat = bouy_df.bouy_longitude.mean(), bouy_df.bouy_latitude.mean()
            print(file_name, bouy, bouy_lon, bouy_lat, len(bouy_df), bouy_df.index)
            
            bouy_atrack, bouy_xtrack = sar_ds.ll2coords(bouy_lon, bouy_lat)        

            atrack_in_range = (0 <= bouy_atrack - dist['atrack']) and \
                            (bouy_atrack + dist['atrack'] <= sar_ds.dataset.atrack[-1].values)
            
            xtrack_in_range = (0 <= bouy_xtrack - dist['xtrack']) and \
                            (bouy_xtrack + dist['xtrack'] <= sar_ds.dataset.xtrack[-1].values)
            
            if not (atrack_in_range and xtrack_in_range):
                print('2x2km not possible around coord')
                continue
            
            small_sar = sar_ds.dataset.sel(
                atrack=slice(bouy_atrack - dist['atrack'], bouy_atrack + dist['atrack'] - 1),
                xtrack=slice(bouy_xtrack - dist['xtrack'], bouy_xtrack + dist['xtrack'] - 1)
            )
                      
            if np.any(small_sar.land_mask):
                print('land in image', i)
                continue
            
            del small_sar['spatial_ref']
            
            out_name = f'{name_counter}.nc'    
            
            record_dict['out_name'].append(out_name)
            record_dict['sar_name'].append(img_name)
            record_dict['bouy_name'].append(bouy)
            record_dict['lat'].append(bouy_lat)
            record_dict['lon'].append(bouy_lon)
            record_dict['index_in_survey_df'].append(str(set(bouy_df.index)))
            
            extract_ds = xr.combine_by_coords([
                #small_sar.digital_number,
                #small_sar.time,
                #small_sar.xtrackSpacing,
                #small_sar.atrackSpacing,
                #small_sar.land_mask,
                small_sar.longitude.astype('float32'),
                small_sar.latitude.astype('float32'), 
                small_sar.sigma0.astype('float32')
            ])

            for k, v in record_dict.items(): 
                print(k, v[-1])
                extract_ds.attrs[k] = v[-1]
            
            extract_ds.to_netcdf(path=out_dir+out_name)
                
            name_counter += 1

record_df = pd.DataFrame(data=record_dict)
record_df.to_csv(out_dir+'sar_segment_table.csv')
