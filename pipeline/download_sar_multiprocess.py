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
import traceback
import pickle
from itertools import repeat
from tqdm import tqdm
from zipfile import ZipFile
import os
import sys
import tifffile as tif
import multiprocessing
import xarray as xr
import xsar

out_dir = '/data/exjobb/sarssw/sar_multiprocess3/'
tmp_dir = '/data/exjobb/sarssw/tmp/'
bouy_survey_path = '/home/sarssw/filip/sarssw/bouy_survey/1h_survey/result_df'
existing_img_dir = '/data/exjobb/sarssw/sar'

num_processes = 10
handle_url_max_retries = 3
box_size = 2000 # 2km

#Determines the offsets for each subimage, the key is the subimage index and 
#the tuple specifies the proportion of the stride length in line and sample direction in proportion to the image length
#compared to an image with offset 0, a factor of 1 results in exactly no overlap, 0.5 in 50% overlap
subimage_prop_offsets = {
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

@contextlib.contextmanager
def sar_download_check_local(url):
    """Downloads the SAR images in a temporary directory, 
    Checks if the file exists localy in existing_img_dir and in that case uses that file"""
    #if files is found locally
    safe_name = url.split('/')[-1].split('.')[0] + '.SAFE'
    if os.path.isdir(existing_img_dir) and safe_name in os.listdir(existing_img_dir):
        #print(f'Found {url} locally')
        yield os.path.join(existing_img_dir, safe_name)

    else:
        #print(f'Downloading {url}')
        with tempfile.TemporaryDirectory(dir = tmp_dir) as tmp_dir_local:
            zip_name = url.split('/')[-1]

            #If we have to donwload the file form asf
            asf.download_url(url=url, path=tmp_dir_local)

            #Unzip to tmp folder
            with ZipFile(os.path.join(tmp_dir_local, zip_name)) as zf:
                zf.extractall(tmp_dir_local)

            yield os.path.join(tmp_dir_local, zip_name.split('.')[0] + '.SAFE')
        
def create_subimages(sar_name, bouy, bouy_df, sar_ds, dist):
    "Extracts the subimages from one downloaded SAR image"
    #print('Handling bouy:', bouy, 'in sar image', sar_name)

    bouy_name = bouy.split('.')[0]
    bouy_lon, bouy_lat = bouy_df.bouy_longitude.mean(), bouy_df.bouy_latitude.mean()
    bouy_line, bouy_sample = sar_ds.ll2coords(bouy_lon, bouy_lat)
    sar_xr = sar_ds.dataset

    #Return when the bouy is not found within the image,
    #Can happen for images at the antimeridian
    if any(np.isnan([bouy_line, bouy_sample])):
        print(f"Skipping {bouy} since it is not found inside the image {sar_name}")
        return
    
    #Subimage limits as subimage_index:(line_min_offset, line_offset_center, line_max_offset, sample_min_offset, sample_offset_center, sample_max_offset)
    subimage_abs_offsets = {}
    for subimage_index, (line_prop_offset, sample_prop_offset) in subimage_prop_offsets.items():
        line_offset_center = int(bouy_line + (line_prop_offset * dist['line']))
        line_min_offset = line_offset_center - dist['line']
        line_max_offset = line_offset_center + dist['line'] - 1
        line_in_range = (0 <= line_min_offset) and (line_max_offset <= sar_xr.line[-1].values)

        sample_offset_center = int(bouy_sample + (sample_prop_offset * dist['sample']))
        sample_min_offset = sample_offset_center - dist['sample']
        sample_max_offset = sample_offset_center + dist['sample'] - 1
        sample_in_range = (0 <= sample_min_offset) and (sample_max_offset <= sar_xr.sample[-1].values)

        if line_in_range and sample_in_range:
            subimage_abs_offsets[subimage_index] = (line_min_offset, line_offset_center, line_max_offset, sample_min_offset, sample_offset_center, sample_max_offset)

    #If we have no valid subimages we return
    if len(subimage_abs_offsets) == 0:
        return

    #Get limits for the offset
    line_abs_min_offset, sample_abs_min_offset = [min([t[i] for t in subimage_abs_offsets.values()]) for i in [0,3]] #0 and 3 is the index for min line and sample
    line_abs_max_offset, sample_abs_max_offset = [max([t[i] for t in subimage_abs_offsets.values()]) for i in [2,5]] #2 and 5 is the index for max line and sample

    #Create larger subimage from which all the smaller are extracted
    large_subimage = sar_xr[['sigma0', 'incidence', 'velocity', 'land_mask', 'time']].sel(
                line=slice(line_abs_min_offset, line_abs_max_offset),
                sample=slice(sample_abs_min_offset, sample_abs_max_offset)
            )
    
    del large_subimage['spatial_ref']

    large_subimage = xr.combine_by_coords([
        large_subimage.sigma0,
        large_subimage.incidence,
        large_subimage.velocity,
        large_subimage.land_mask,
        large_subimage.time,
    ])

    #Save some of the metadata from the original
    save_attrs = ['safe', 'swath', 'platform', 'orbit_pass', 'product', 'platform_heading']
    for attr in save_attrs:
        large_subimage.attrs[attr] = sar_xr.attrs[attr]
    
    #Add a few extra metadata common for all subimages
    for attr_name, attr_value in [('sar_name', sar_name), ('bouy_name', bouy_name), ('polarisations', ' '.join(large_subimage['pol'].values))]:
        large_subimage.attrs[attr_name] = attr_value
    
    #Load large_subimg to memory
    large_subimage.load(scheduler='threads')

    #Iterate over all subimages, extract and save
    for subimage_index, (line_min_offset, line_offset_center, line_max_offset, sample_min_offset, sample_offset_center, sample_max_offset) in subimage_abs_offsets.items():

        subimage = large_subimage.sel(
            line=slice(line_min_offset, line_max_offset),
            sample=slice(sample_min_offset, sample_max_offset)
        )

        #Skip if there is land within the image or if there are nans due to subimages over edges of the original image
        if np.any(np.isnan(subimage.sigma0.values)) or \
           np.any(np.isnan(subimage.incidence.values)) or \
           np.any(subimage.land_mask.values):
            continue

        #Save coordinates for center of subimage
        img_center_lon, img_center_lat = sar_ds.coords2ll(line_offset_center, sample_offset_center)

        #add subimage specific metadata
        for attr_name, attr_value in [
                ('longitude',img_center_lon),
                ('latitude',img_center_lat),
                ('time',str(np.array([subimage.sel(line=line_offset_center, method='nearest')['time'].values], dtype="datetime64[ns]")[0])),
                ('subimage_index', subimage_index)]:
            subimage.attrs[attr_name] = attr_value

        #Drop landmask & time data variables
        subimage = subimage.drop_vars(['land_mask', 'time'])

        out_path = os.path.join(out_dir, f'{sar_name}-{bouy_name}-{subimage_index}.nc')

        subimage.to_netcdf(path=out_path)


def handle_url(args):
    "The main logic for handling one url"
    url, bouy_survey_df = args
    #print('Handling url:', url)

    try:
        with sar_download_check_local(url) as safe_path:
            sar_meta = xsar.Sentinel1Meta(safe_path)
            sar_ds = xsar.Sentinel1Dataset(sar_meta)

            sar_name = url.split('/')[-1].split('.')[0]
            dist = {
                'line': int(np.round(box_size / 2 / sar_meta.pixel_line_m)),
                'sample': int(np.round(box_size / 2 / sar_meta.pixel_sample_m))
            }
        
            bouy_df_part = bouy_survey_df[bouy_survey_df.sar_url == url]
            
            for bouy, bouy_df in bouy_df_part.groupby('bouy_file_name'):
                create_subimages(sar_name, bouy, bouy_df, sar_ds, dist)

    #Print what file was interrupted
    except KeyboardInterrupt:
        print(f'Interrupted the handling of the url: {url}')
        sys.exit(1)

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
            traceback.print_exc()

            if tries >= handle_url_max_retries:
                print(f'skiping {url}')
                return
    
def multiprocess_urls(urls, bouy_survey_df):
    "Multi processed handling of the urls"
    if num_processes == 1:
        for url in tqdm(urls):
            handle_url_with_retry((url, bouy_survey_df))

    else:
        with multiprocessing.Pool(num_processes) as pool, multiprocessing.Manager() as manager:
            try:
                shutdown = manager.Event()
                for _ in tqdm(pool.imap_unordered(handle_url_with_retry, zip(urls, repeat(bouy_survey_df, len(urls)))), total=len(urls)):
                    pass

            except KeyboardInterrupt:
                shutdown.set()
                print('Parent keyboard interrupt caught, shutdown (waiting)')
                pool.terminate()
                print('Shutdown done')
                sys.exit('Keyboard interrupt')

def main():
    #Read survey data
    with open(bouy_survey_path,'rb') as f_r:
        bouy_survey_df = pickle.load(f_r)

    #Extract urls for SAR images, sorting after most number of bouys contained
    all_urls = bouy_survey_df.groupby('sar_url').count().\
        sort_values(by='bouy_file_name', ascending=False).index.to_numpy()

    #Create out_dir adn tmp_dir in case they do not already exists
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    
    filenames = filter(lambda fn: fn.endswith('.nc'), os.listdir(out_dir))
    processed_urls = {fn.split('-')[0] for fn in filenames}
    urls = [url for url in all_urls if url.split('/')[-1].split('.')[0] not in processed_urls]
    print(f'urls found {len(all_urls)}')
    print(f'urls already processed {len(processed_urls)}')
    print(f'urls to handle {len(urls)}')
    
    multiprocess_urls(urls, bouy_survey_df)

if __name__ == "__main__":
    main()