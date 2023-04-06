#!/usr/bin/env python3

"""
This program downloads the sar images found in the bouy survey data result dataframe found in the folder specified by the variable bouy_survey_dir.
It then extracts subimages for the bouy(s) contained within that image and save them to the folder specified by out_dir.
This program assumes that you have your cridentials to datapool.asf.alaska.edu in the ~/.netrc file
"""

import asf_search as asf
import numpy as np
import contextlib
import tempfile
import pickle
from itertools import repeat
from tqdm import tqdm
from zipfile import ZipFile
import os
import tifffile as tif
import multiprocessing
import xsar

out_dir = '/data/exjobb/sarssw/sar_multiprocess2/'
tmp_dir = '/data/exjobb/sarssw/tmp'
bouy_survey_dir = '/home/sarssw/axel/sarssw/bouy_survey/1h_survey'
result_df_fn = 'result_df'

num_processes = 10
handle_url_max_retries = 3
box_size = 2000 # 2km

crop_offsets = {
    8:( .5, -.5), 1:( .5, 0), 2:( .5, .5),
    7:(  0, -.5), 0:(  0, 0), 3:(  0, .5), 
    6:(-.5, -.5), 5:(-.5, 0), 4:(-.5, .5)
}

#Currently using .netrc file instead, to change to manual input uncomment these and adjust sar_download accordingly specifying the session argument in asf.download_url
#username = input('Username:')
#password = getpass.getpass('Password:')
#session = asf.ASFSession().auth_with_creds(username=username, password=password)

@contextlib.contextmanager
def sar_download(url):
    "Downloads the SAR images in a temporary directory"
    #print('Downloading:', url)
    with tempfile.TemporaryDirectory(dir = tmp_dir) as tmp_dir_local:    
        asf.download_url(url=url, path=tmp_dir_local)       

        zip_name = url.split('/')[-1]   
        
        with ZipFile(os.path.join(tmp_dir_local, zip_name)) as zf:
            zf.extractall(tmp_dir_local)

        yield os.path.join(tmp_dir_local, zip_name.split('.')[0] + '.SAFE')



def create_subimages(sar_name, bouy, bouy_df, sar_ds, dist):
    "Extracts the subimages from one downloaded SAR image"
    #print('Handling bouy:', bouy)
    bouy_name = bouy.split('.')[0]
    bouy_lon, bouy_lat = bouy_df.bouy_longitude.mean(), bouy_df.bouy_latitude.mean()
    
    bouy_atrack, bouy_xtrack = sar_ds.ll2coords(bouy_lon, bouy_lat)

    #Return when the bouy is not found within the image,
    #Can happen for images at the antimeridian
    if any(np.isnan([bouy_atrack, bouy_xtrack])):
        print(f"Skipping {bouy} since it is not found inside the image {sar_name}")
        return
    
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

        #Skip if there is land within the image or if there are nans due to subimages over edges of the original image
        if np.any(small_sar.land_mask) or np.any(np.isnan(small_sar.sigma0.values)): continue
            
        img_center_lon, img_center_lat = sar_ds.coords2ll(offset_atrack, offset_xtrack)

        out_path = os.path.join(out_dir, f'{sar_name}-{bouy_name}-{crop_index}.tif')
        tif.imwrite(out_path,
                    small_sar.sigma0.values,
                    metadata={
                        'sar_name':sar_name,
                        'bouy_name':bouy_name,
                        'subimage_index':crop_index,
                        'pol':list(small_sar['pol'].values),
                        'time':str(np.array([small_sar.sel(atrack=offset_atrack, method='nearest')['time'].values], dtype="datetime64[ns]")[0]),
                        'lon':img_center_lon,
                        'lat':img_center_lat,
                    })

def handle_url(args):
    "The main logic for handling one url"
    url, bouy_survey_df = args
    #print('Handling url:', url)

    with sar_download(url) as safe_path:
        # atrack = line, xtrack = sample
        sar_meta = xsar.Sentinel1Meta(safe_path)
        sar_ds = xsar.Sentinel1Dataset(sar_meta)

        sar_name = url.split('/')[-1].split('.')[0]
        dist = {
            'atrack': int(np.round(box_size / 2 / sar_meta.pixel_atrack_m)),
            'xtrack': int(np.round(box_size / 2 / sar_meta.pixel_xtrack_m))
        }
    
        bouy_df_part = bouy_survey_df[bouy_survey_df.sar_url == url]
        
        for bouy, bouy_df in bouy_df_part.groupby('bouy_file_name'):
            create_subimages(sar_name, bouy, bouy_df, sar_ds, dist)

def handle_url_with_retry(args):
    "Wraps exception handling around the handle_url"
    url, bouy_survey_df = args

    for tries in range(1, handle_url_max_retries + 1):
        if tries > 1:
            print(f'Retrying (try {tries}) URL {url}')
        try:
            handle_url(args)
            break

        except Exception as e:
            print(f'Tried handling url {url} {tries} times and failed\nexception:', e)
            if tries >= handle_url_max_retries:
                print(f'skiping {url}')
                return
    
def multiprocess_urls(urls, bouy_survey_df):
    "Multi processed handling of the urls"
    with multiprocessing.Pool(num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(handle_url_with_retry, zip(urls, repeat(bouy_survey_df, len(urls)))), total=len(urls)):
            pass

def main():
    #Read survey data
    with open(os.path.join(bouy_survey_dir, result_df_fn),'rb') as f_r:
        bouy_survey_df = pickle.load(f_r)

    #Extract urls for SAR images, sorting after most number of bouys contained
    all_urls = bouy_survey_df.groupby('sar_url').count().\
        sort_values(by='bouy_file_name', ascending=False).index.to_numpy()


    #Create out_dir adn tmp_dir in case they do not already exists
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    

    filenames = filter(lambda fn: fn.endswith('.tif'), os.listdir(out_dir))
    processed_urls = {fn.split('-')[0] for fn in filenames}
    print(len(all_urls), len(processed_urls))
    urls = [url for url in all_urls if url.split('/')[-1].split('.')[0] not in processed_urls]
    print(len(urls))
    
    multiprocess_urls(urls, bouy_survey_df)

if __name__ == "__main__":
    main()