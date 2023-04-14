#!/usr/bin/env python3

"""
This module is used to extract the complete dataset with features and labels from the folder with sar subimages
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import math
import xarray as xr

import tifffile as tif
from collections import defaultdict
from itertools import islice

from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte
from tqdm import tqdm


class Var_results:
    "Used in load_label_df to keep track of wave height and wind speed results"
    def __init__(self):
        self.found = False
        self.source = ''
        self.value = 0
        self.lon = 0
        self.lat = 0
        self.time = np.datetime64('1970-01-01T00:00')

    def __repr__(self):
        return f"""
            found {self.found}
            value {self.value}
            source {self.source}
            lon {self.source}
            lat {self.lat}
            time {self.time}
            """
    
    def to_list(self):
        return [
            self.value,
            self.source,
            self.lon,
            self.lat,
            self.time,
        ]

#to adjust the wind speed acording to the wind profile power law
#found at https://en.wikipedia.org/wiki/Wind_profile_power_law
#using exponent parameter of 0.11
#Depth is the negative height
def wspd_power_law(wspd, depth):
    #If the height is 0 assume 10 meters
    if depth == 0:
        return wspd_power_law(wspd, 10)
    
    return wspd * (10/(-1*depth))**0.11
        
def load_features_df(sar_paths, svc_file):
    "Extracts the features in a dictionar"
    with open(svc_file, 'rb') as f: svc = pickle.load(f)
    feature_dict = defaultdict(list)
    #metadata_dict = defaultdict(list)
    
    print('Calculating features')
    for file_path in tqdm(sar_paths):
        if not file_path.endswith('.nc'): continue
        
        xds =  xr.open_dataset(file_path)
        #metadata_dict['file_name'].append(file_name)
        #for metadata_key, metadata_value in tif_img.shaped_metadata[0].items():
        #    metadata_dict[metadata_key].append(metadata_value)
        
        for pol, v in zip(xds.sigma0.coords['pol'].values, xds.sigma0.values):
            if np.isnan(v).any(): continue

            ubyte = img_as_ubyte((v - v.min()) / (v.max() - v.min()))
            glcm = graycomatrix(ubyte, distances=[10], angles=[0], levels=256,
                                symmetric=True, normed=True)

            all_glcm_types = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
            sar_glcm = {gt:graycoprops(glcm, gt)[0, 0] for gt in all_glcm_types}

            glcm_types =  ['homogeneity', 'dissimilarity', 'correlation']
            hom_test = svc.predict([[sar_glcm[gt] for gt in glcm_types]])[0] == 0
            
            #filename and polarization for given image
            feature_dict['file_name'].append(file_path.split('/')[-1])
            feature_dict['pol'].append(pol)
            
            #metadata from image
            for metadata_key, metadata_value in xds.attrs.items():
                if metadata_key not in ['pol']:
                    feature_dict[metadata_key].append(metadata_value)
            
            #bool for if image is homogenous or not
            feature_dict['hom_test'].append(hom_test)
            
            #all features extracted from glcm
            for glcm_type, glcm_value in sar_glcm.items():
                feature_dict[glcm_type].append(glcm_value)
            
            #features from sigma0 aggregations
            feature_dict['sigma_mean'].append(v.mean())
            feature_dict['sigma_var'].append(v.var())
            feature_dict['sigma_mean_over_var'].append(v.mean() / v.var())
            feature_dict['sigma_min'].append(v.min())
            feature_dict['sigma_max'].append(v.max())
            feature_dict['sigma_range'].append(v.max() - v.min())

    return pd.DataFrame(feature_dict)

def load_labels_df(bouy_survey_path, swh_model_path, wspd_model_path, sar_bouy_df):
    """
    This function loads the labels for files indexed with sar_name and bouy_name, those pairs are given as columns in sar_bouy_df
    
    Parameters
    bouy_survey_path: path to the survey dataframe
    swh_model_path: path to the wave height model
    wspd_model_path: path to the wind speed model
    
    Returns
    A dataframe with the labels
    """

    #Load the bouy survey dataframe
    with open(bouy_survey_path,'rb') as f_r:
        bouy_survey_df = pickle.load(f_r)
        
    #Reindex to enable faster lookup
    bouy_survey_df['sar_name'] = bouy_survey_df['sar_url'].apply(lambda row: row.split('/')[-1].split('.')[0])
    bouy_survey_df['bouy_name'] = bouy_survey_df['bouy_file_name'].apply(lambda row: row.split('.')[0])
    bouy_survey_df = bouy_survey_df.set_index(['sar_name', 'bouy_name']).sort_index()

    #wave height model
    print("Loading wave height model")
    SWH_model = xr.open_dataset(swh_model_path)

    #Wind speed model
    print("Loading wind speed model")
    WSPD_model = xr.open_dataset(wspd_model_path)

    #Configure and program how the models work
    
    #Program how the labels are loaded from the survey dataframe
    #His saves wave high as is and hight adjusts wind speed to 10m using the power law
    suvey_value_functions = {
        'SWH': lambda row: row['bouy_variable_value'],
        'WSPD': lambda row: wspd_power_law(row['bouy_variable_value'], row['bouy_depth']),
    }
    
    var_list = ['SWH', 'WSPD']
    var_names = {
        'SWH': ['VHM0', 'VAVH'],
        'WSPD': ['WSPD'],
    }

    models = {
        'SWH':SWH_model,
        'WSPD':WSPD_model,
    }

    model_coords_columns =  {
        'SWH': {'time':'time', 'longitude':'longitude', 'latitude':'latitude'},
        'WSPD': {'time':'time', 'longitude':'lon', 'latitude':'lat'},
    }

    model_value_functions = {
        'SWH': (lambda row: float(row['swh'])),
        'WSPD': (lambda row: math.sqrt(row['northward_wind']**2 + row['eastward_wind']**2)),
    }

    #Create dataframe for labels
    labels_df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in {
                'sar_name':str,
                'bouy_name':str,
                'SWH_value': float,
                'SWH_source':str,
                'SWH_lon':float,
                'SWH_lat':float,
                'SWH_time':np.dtype('<M8[ns]'), #np.datetime64
                'WSPD_value': float,
                'WSPD_source':str,
                'WSPD_lon':float,
                'WSPD_lat':float,
                'WSPD_time':np.dtype('<M8[ns]'), #np.datetime64
            }.items()})
    
    #Filter the sar_bouy_df (index) dataframe to only include the relevant columns
    sar_bouy_df = sar_bouy_df[['sar_name', 'bouy_name']]

    print('Collecting labels')
    for i, (sar_name, bouy_name) in tqdm(sar_bouy_df.iterrows(), total=sar_bouy_df.shape[0]):
        result_vars = {
            'SWH':Var_results(),
            'WSPD':Var_results(),
        }

        #Extract value(s) form survey
        #The result can contain multiple rows, one for each measurement (wind speed, wave height as VHM0 and/or VAVH)
        survey_results = bouy_survey_df.loc[(sar_name, bouy_name)]
        #save longitude, latitude and time for eventual model search
        lon, lat, start_time, end_time = survey_results.iloc[0][['bouy_longitude', 'bouy_latitude', 'sar_start_time', 'sar_stop_time']]
        time = start_time+(end_time-start_time)/2

        #Save value(s) from survey
        for label, result in survey_results.iterrows():
            for var in var_list:
                if (result['bouy_variable_name'] in var_names[var]) and (not result_vars[var].found):
                    result_vars[var].found = True
                    result_vars[var].source = 'bouy'
                    result_vars[var].value = suvey_value_functions[var](result)
                    result_vars[var].lon = result['bouy_longitude']
                    result_vars[var].lat = result['bouy_latitude']
                    result_vars[var].time = result['bouy_time']

        #Complete missing value form model
        for var in var_list:
            if not result_vars[var].found:
                model_lon = model_coords_columns[var]['longitude']
                model_lat = model_coords_columns[var]['latitude']
                model_time = model_coords_columns[var]['time']

                model_result = models[var].interp({
                    model_lon:xr.DataArray([lon], dims='unused_dim'),
                    model_lat:xr.DataArray([lat], dims='unused_dim'),
                    model_time:xr.DataArray([time], dims='unused_dim')},
                    method='linear').to_dataframe().iloc[0]

                result_vars[var].found = True
                result_vars[var].source = 'model'
                result_vars[var].value = model_value_functions[var](model_result)
                result_vars[var].lon = model_result[model_lon]
                result_vars[var].lat = model_result[model_lat]
                result_vars[var].time = model_result[model_time]

        #Append result to the dataframe
        labels_df.loc[len(labels_df.index)] = [sar_name, bouy_name] + result_vars['SWH'].to_list() + result_vars['WSPD'].to_list()

    return labels_df

def load_features_labels_df(sar_dir, svc_file, bouy_survey_fn, swh_model_fn, wspd_model_fn):
    "This function is load s a dataframe containing both features and labels"

    features_df = load_features_df(sar_dir, svc_file)
    sar_bouy_df = features_df[['sar_name', 'bouy_name']].drop_duplicates()
    labels_df = load_labels_df(bouy_survey_fn, swh_model_fn, wspd_model_fn, sar_bouy_df)

    return features_df.set_index(['sar_name', 'bouy_name']).sort_index().join(labels_df.set_index(['sar_name', 'bouy_name']).sort_index())


if __name__ == "__main__":
    sar_dir = '/data/exjobb/sarssw/sar_multiprocess/'   
    svc_file = './out/homogenity_svc.pkl'
    feature_df = load_features_df(sar_dir, svc_file)
    print(feature_df)