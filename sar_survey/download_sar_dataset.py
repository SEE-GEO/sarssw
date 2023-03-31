import asf_search as asf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from collections import defaultdict
import itertools
import contextlib
import tempfile

import pickle
from tqdm import tqdm
from zipfile import ZipFile
import os
import getpass
import tifffile as tif

import xsar
import xarray as xr

out_dir = '/data/exjobb/sarssw/sar_dataset/'
write_folder = '/home/sarssw/axel/sarssw/bouy_survey/1h_survey'
result_df_fn = 'result_df'

with open(os.path.join(write_folder, result_df_fn),'rb') as f_r:
    shore_survey_df = pickle.load(f_r)

username = input('Username:')
password = getpass.getpass('Password:')
session = asf.ASFSession().auth_with_creds(username=username, password=password)

@contextlib.contextmanager
def sar_download(url):
    with tempfile.TemporaryDirectory(dir = '/data/exjobb/sarssw/') as tmp_dir:    
        asf.download_url(url=url, path=tmp_dir, session=session)       
        zip_name = url.split('/')[-1]   
        
        with ZipFile(os.path.join(tmp_dir, zip_name)) as zf:
            zf.extractall(tmp_dir)

        yield os.path.join(tmp_dir, zip_name.split('.')[0] + '.SAFE')

all_urls = shore_survey_df.groupby('sar_url').count().\
       sort_values(by='bouy_file_name', ascending=False).index.to_numpy()

filenames = filter(lambda fn: fn.endswith('.tif'), os.listdir(out_dir))
processed_urls = {fn.split('-')[0] for fn in filenames}
print(len(all_urls), len(processed_urls))
urls = [url for url in all_urls if url.split('/')[-1].split('.')[0] not in processed_urls]
print(len(urls))

box_size = 2000 # 2km

crop_offsets = {
    8:( .5, -.5), 1:( .5, 0), 2:( .5, .5),
    7:(  0, -.5), 0:(  0, 0), 3:(  0, .5), 
    6:(-.5, -.5), 5:(-.5, 0), 4:(-.5, .5)
}

for url in tqdm(urls, total = len(urls)):
    print('Handling url:', url)
    with sar_download(url) as safe_path:
        sar_name = url.split('/')[-1].split('.')[0]
        
        # atrack = line, xtrack = sample
        sar_meta = xsar.Sentinel1Meta(safe_path)
        sar_ds = xsar.Sentinel1Dataset(sar_meta)
        dist = {
            'atrack': int(np.round(box_size / 2 / sar_meta.pixel_atrack_m)),
            'xtrack': int(np.round(box_size / 2 / sar_meta.pixel_xtrack_m))
        }
        
        shore_df_part = shore_survey_df[shore_survey_df.sar_url == url]
        
        for bouy, bouy_df in shore_df_part.groupby('bouy_file_name'):
            print('Handling bouy:', bouy)
            bouy_name = bouy.split('.')[0]
            bouy_lon, bouy_lat = bouy_df.bouy_longitude.mean(), bouy_df.bouy_latitude.mean()
            
            bouy_atrack, bouy_xtrack = sar_ds.ll2coords(bouy_lon, bouy_lat)
            
            for crop_index, (atrack_offset, xtrack_offset) in crop_offsets.items():
                offset_atrack = int(bouy_atrack + (atrack_offset * dist['atrack']))
                offset_xtrack = int(bouy_xtrack + (xtrack_offset * dist['xtrack']))

                atrack_in_range = (0 <= offset_atrack - dist['atrack']) and \
                                (offset_atrack + dist['atrack'] <= sar_ds.dataset.atrack[-1].values)

                xtrack_in_range = (0 <= offset_xtrack - dist['xtrack']) and \
                                (offset_xtrack + dist['xtrack'] <= sar_ds.dataset.xtrack[-1].values)

                if not (atrack_in_range and xtrack_in_range): continue

                small_sar = sar_ds.dataset.sel(
                    atrack=slice(offset_atrack - dist['atrack'], offset_atrack + dist['atrack'] - 1),
                    xtrack=slice(offset_xtrack - dist['xtrack'], offset_xtrack + dist['xtrack'] - 1)
                )

                if np.any(small_sar.land_mask): continue
                    
                out_path = os.path.join(out_dir, f'{sar_name}-{bouy_name}-{crop_index}.tif')
                tif.imwrite(out_path, small_sar.sigma0.values)
