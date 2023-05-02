import numpy as np
import pandas as pd
import xarray as xr
import os
import asf_search as asf
from collections import defaultdict
import datetime
import pickle
from tqdm import tqdm
from functools import reduce
import itertools
from shapely import Point, LineString, Polygon, MultiPolygon
import operator
import lxml
from pykml.factory import KML_ElementMaker as KML
import cartopy

#Extracts data from the dataset ds within the time_filter (tuple or timespan) interval for the 
#variable var_name found in the deph range deph_range in meters, positive is under water, negative above water
#It can be either a tuble (min,max) or a value it needs to equal
#Quality controll is made for position, deph, time, and the variable
#Note depth is the coordinate index while deph (without t) is the actual depth in meters 
def valid_data_extraction(ds, time_filter, var_name, deph_range):
    if var_name not in ds.data_vars:
        raise ValueError(var_name, ' Not found')

    #Add longitude, latidude and position_qc as variables indexed by time,depth as all other variables
    TIME = ds['TIME'].values
    DEPTH = ds['DEPTH'].values
    n_DEPTHS = len(DEPTH)

    dataset_columns = {
        'LONG':ds['LONGITUDE'],
        'LAT':ds['LATITUDE'],
        'POS_QC':ds['POSITION_QC'],
    }

    ds_pos = xr.Dataset(
        data_vars=
        {k:(
            ["TIME", 'DEPTH'],
            np.repeat(np.reshape(v.values, (-1,1)), n_DEPTHS, axis=1),
            v.attrs,
        )for (k,v) in dataset_columns.items()},
        coords=dict(
            TIME=TIME,
            DEPTH=DEPTH,
        )
    ).drop_vars('DEPTH')
    ds = xr.merge([ds.drop_dims(['LATITUDE', 'LONGITUDE', 'POSITION']), ds_pos])
    
    #Filter for time of interest
    if type(time_filter) is tuple:
        ds = ds.sel(TIME=slice(time_filter[0], time_filter[1]))
    else:
        ds = ds.sel(TIME=time_filter)
    
    #Filter only avalible columns
    colum_names = [var_name]
    colum_names_qc = [var_name + '_QC']
    
    #Add fixed columns
    colum_names.extend(['LONG', 'LAT', 'DEPH'])
    colum_names_qc.extend(['DEPH_QC'])
    time_pos_qc = ['TIME_QC', 'POS_QC']
    
    #Filter for columns of interest
    ds = ds[colum_names + colum_names_qc + time_pos_qc]

    df = ds.to_dataframe()
    
    QC_good = [1.0, 7.0]
    #QC control for time and pos uses all of these values according to https://doi.org/10.13155/59938
    QC_time_pos_good = [1, 2, 5, 7, 8]
    
    #Filter the variable and depth for good quality data 
    filter_qc = [df[c_qc].isin(QC_good) for c_qc in colum_names_qc]
    #Filter for good time and pos 
    filter_qc.extend([df[c_qc].isin(QC_time_pos_good) for c_qc in time_pos_qc])
    #Add filter for deph value
    if type(deph_range) == tuple:
        filter_qc.append((deph_range[0] <= df['DEPH']) & (df['DEPH'] <= deph_range[1]))
    else:
        filter_qc.append(df['DEPH'] == deph_range)
    #Element-wise AND the filter
    filter_qc = reduce(operator.and_, filter_qc)
    df = df[filter_qc][colum_names]

    #Remove any duplicated measurements on different depths
    #To do so, first remove the depth from the index and then filter for unique time index by keeing the first
    df = df.reset_index('DEPTH')
    df = df[~df.index.duplicated(keep='first')]

    return df


# Searches for SAR images that overlaps with the data in file according to the filters specified by the arguments:

# file: specifies the file name of the bouy data

# data_dir: is the directory

# start_date & end_date: are the time filters for the data

# variables_dephs: The keys are the variable to extract data for while the values are the accepted depth ranges 
# (in meters) for the corresponding variable negative values are depth under surface while positive is above 

# max_time_diff_s: Is the absolute time difference between image and bouy observation in seconds

# land_multipolygon: is the shapely multipolygon object that describes the land and close to shore map of earth
# observations from these areas are filtered

# result_df: is the dataframe where the result is saved, if it is passed as None it is automatically created with the correct columns

# The data is also quiality controlled and only rows containing valid in regards to the variable, time, position and depth are accepted
def search_file(file, data_dir, start_date, end_date, variables_dephs, max_time_diff_s, land_multipolygon, result_df):
    #Conditionally create the result dataframe
    if result_df is None:
        result_df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in {
            'bouy_file_name':str,
            'bouy_longitude':float,
            'bouy_latitude':float,
            'bouy_time':np.dtype('<M8[ns]'), #np.datetime64
            'bouy_depth_index':int, #The depth index
            'bouy_depth':float, #The deph (depth) value in meters, positive is under water
            'bouy_variable_name':str,
            'bouy_variable_value':float,
            'sar_url':str,
            'sar_mode':str,
            'sar_polarization':str,
            'sar_platform':str,
            'sar_start_time':np.dtype('<M8[ns]'), #np.datetime64
            'sar_stop_time':np.dtype('<M8[ns]'), #np.datetime64
            'sar_coordinates':object, #List
        }.items()})

    #Load the data from the file
    file_path = os.path.join(data_dir, file)
    xar = xr.open_dataset(file_path)# , engine='scipy')
    xar_vars = list(xar.data_vars)

    #Clear any eventual prior asf search result
    asf_results = None
    
    #Filter for variables that exist in the data
    common_variables = set(variables_dephs.keys()).intersection(xar_vars)
    
    if len(common_variables) == 0:
        #print(file + " does not have any of the variables " + str(variables))
        return result_df
    
    for var in common_variables:
        try:
            #extract dataframe of the correct timeinterval and variable 
            df = valid_data_extraction(xar, (start_date, end_date), var, variables_dephs[var])
        except Exception as e:
            print(file, ': data extraction failed with:', e)
            continue
        
        if df.empty:
            #print(file + " does not have any valid data for this timeperiod")
            continue
            
        #Only search once (assume similar results in regards to min and max longitude and latitude for each variable)
        if asf_results is None:
            #extract long and latitude
            long_min, lat_min = df[['LONG', 'LAT']].min()
            long_max, lat_max = df[['LONG', 'LAT']].max()

            #Create geographical search restriction
            coord_points = list(itertools.product(set([long_min, long_max]), set([lat_min, lat_max])))

            #For each possible length of coord_points set the prefix ans dufix for the asf search
            #Also create a shapely geometry to check for overlap with the land_multipolygon
            if len(coord_points) == 1:
                geo_prefix = 'POINT('
                geo_suffix = ')'
                geo_obj = Point(*coord_points)
            elif len(coord_points) == 2:
                #raise Exception("First time we see only changes in one coordinate, debug and make sure it works porpery")
                geo_prefix = 'LINESTRING('
                geo_suffix = ')'
                geo_obj = LineString(coord_points)
            elif len(coord_points) == 4:
                #Double parenthesis was given from asf website export search as python function
                geo_prefix = 'POLYGON(('
                geo_suffix = '))'
                #Untangle the polygon itersection
                coord_points[2:4]=reversed(coord_points[2:4])
                #Form closed polygon by adding adding fist point as last
                coord_points.append(coord_points[0])
                geo_obj = Polygon(coord_points)

            #The area that of the locations in the data that is not near the shoreline
            off_shore = geo_obj.difference(land_multipolygon)

            #Check if the whole area of interest is close to shore or overlapps with land
            #in that case return
            if off_shore.is_empty:
                #print(file + " is too close to land")
                return result_df

            #Filter data of df to only include data that does not lie close to shore
            #allow a small distance since points on the edge doeas not overlap according to shapely.overlaps 
            df_shore_filter =  df.apply(lambda row: Point([row['LONG'], row['LAT']]).distance(off_shore), axis=1) <= 0.00001
            df = df[df_shore_filter]

            geo_limit = ','.join([str(long) + ' ' + str(lat) for long,lat in coord_points])
            geo_limit = geo_prefix + geo_limit + geo_suffix
            
            #print(geo_limit)

            #Search asf
            options = {
                'intersectsWith': geo_limit,
                'platform': 'SENTINEL-1',
                'instrument': 'C-SAR',
                'start': start_date,
                'end': end_date,
                'processingLevel': [
                    'GRD_HD',
                    'GRD_MD',
                    'GRD_MS',
                    'GRD_HS'
                ],
                'beamSwath': [
                    'IW',
                    'EW',
                ],
            }

            asf_results = asf.search(**options)
            if len(asf_results) != 0:
                #print(len(asf_results), ' results found')
                pass
        
        for asf_result in asf_results:
            #Find mean time of image generation
            asf_result_start = datetime.datetime.fromisoformat(asf_result.properties['startTime'][:-5])
            asf_result_stop = datetime.datetime.fromisoformat(asf_result.properties['stopTime'][:-5])
            asf_result_mean_time = asf_result_start + (asf_result_stop-asf_result_start)/2
        
            #Check the shape type
            if asf_result.geometry['type'] != 'Polygon':
                raise ValueError('The shape of the search result is not Polygon but instead: ' + asf_result.geometry['type'])
            
            #Filter for points that lie within the image
            if geo_prefix != 'POINT(':
                polygon = Polygon(asf_result.geometry['coordinates'][0])
                df_dist_filter =  df.apply(lambda row: Point([row['LONG'], row['LAT']]).distance(polygon), axis=1) <= 0.0001
                df_close = df[df_dist_filter]
                
                #Skip is we have no overlaping points
                if df_close.empty:
                    #print("No geographical overlap")
                    continue
                    
            #No geographical overlap check needed (point search)
            else:
                df_close = df
            
            #Find closest entry in time
            closest_data_row = df_close.iloc[df_close.index.get_indexer([asf_result_mean_time], method='nearest')] #only one row of data

            max_time_diff = datetime.timedelta(seconds=max_time_diff_s)
            
            #Check if it is close enough to be considered overlapping
            if abs(closest_data_row.index[0] - asf_result_mean_time) <= max_time_diff:
                #Save result in the dataframe
                result_df.loc[len(result_df.index)] = [
                    file, #'bouy_file_name':str,
                    closest_data_row['LONG'][0], #'bouy_longitude':float,
                    closest_data_row['LAT'][0], #'bouy_latitude':float,
                    closest_data_row.index[0], #'bouy_time':np.dtype('<M8[ns]'), #np.datetime64
                    closest_data_row['DEPTH'][0], #'bouy_depth_index':int, #The depth index
                    closest_data_row['DEPH'][0], #'bouy_depth':float,
                    var, #'bouy_variable_name':str,
                    closest_data_row[var][0], #'bouy_variable_value':float,
                    asf_result.properties['url'], #'sar_url':str,
                    asf_result.properties['beamModeType'], #'sar_mode':str,
                    asf_result.properties['polarization'].split('+'), #'sar_polarization':str,
                    asf_result.properties['platform'], #'sar_platform':str,
                    asf_result_start, #'sar_start_time':np.dtype('<M8[ns]'), #np.datetime64
                    asf_result_stop, #'sar_stop_time':np.dtype('<M8[ns]'), #np.datetime64
                    asf_result.geometry['coordinates'][0], #'sar_coordinates':list,
                ]
                
            else:
                #print('No time overlap')
                pass
            
        
    return result_df

if __name__ == "__main__":
    data_dir = '/data/exjobb/sarssw/bouy/INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030/MO'
    start_date = '2021-01-01'
    end_date = '2021-12-31'
    write_folder = './1h_survey_2021'
    variables = ['VHM0', 'VAVH', 'WSPD']
    
    #Remember to also configure the asf search option in search_file.
    
    file_filter = [
        'GL_TS_MO_41121.nc', #Flips longitude sign in the middle of the data, from 66 to -66???! resutlts in asf search with over 7000 matches.
    ]
    
    files = set(os.listdir(data_dir)).difference(file_filter)

    variables_deph = {'VHM0':0, 'VAVH':0, 'WSPD':(-30,0)}#['VHM0', 'VAVH', 'WSPD']
    max_time_diff_s = 60*60

    result_df_fn = 'result_df'
    kml_pinmap_fn = 'kml_pinmap'

    #Dataframe where the results is saved, automatically created in search_file
    result_df = None

    #Load and create land multipolygon, buffered (expanded) to limit distance to shore
    land_list = list(cartopy.feature.NaturalEarthFeature('physical', 'land', '50m').geometries())
    polygon_list = []
    for p  in land_list:
        if type(p) == MultiPolygon:
            polygon_list.extend(p.geoms)
        else:
            polygon_list.append(p)
    land_multipolygon = MultiPolygon([p.buffer(0.01) for p in polygon_list])

    # Create search_file function with only necessary parameters for the file loop below
    def search_file_fixed_params(file, result_df):
        return search_file(file, data_dir, start_date, end_date, variables_deph, max_time_diff_s, land_multipolygon, result_df)

    for file in tqdm(files):
        #print('\n',file)
        result_df = search_file_fixed_params(file, result_df)

    #Save resuld_df with pickle

    #Conditionally creates the folder for the result
    os.makedirs(write_folder, exist_ok=True)

    with open(os.path.join(write_folder, result_df_fn),'wb') as f_w:
        pickle.dump(result_df,f_w)


    #Save KML pin maps, one for each variable

    #Total number of datapoints per variable type
    var_total = result_df['bouy_variable_name'].value_counts()
    
    #pandas series for count of unique variable name, file name pairs
    var_file_count = list(result_df.value_counts(subset=['bouy_variable_name', 'bouy_file_name']).items())

    #dataframe for coordinates of unique variable name, file name pairs
    var_file_coord = (result_df.groupby(['bouy_variable_name', 'bouy_file_name'])[['bouy_longitude', 'bouy_latitude']].first())


    #Create separate kml maps for each variable name
    KML_fldrs = {}
    for var_name in np.unique(result_df['bouy_variable_name']):
        KML_fldrs[var_name] = KML.Folder(
            KML.name(var_name + " " + write_folder[write_folder.find('/')+1:]),
            KML.description('Total datapoints: ' + str(var_total[var_name])),
        )

    #iterate over all unique variable name, file name pairs
    for ((var_name, file_name), count) in var_file_count:
        #extract longitude and latitude
        long, lat = var_file_coord.loc[(var_name, file_name)][['bouy_longitude', 'bouy_latitude']]
        
        #Create the pin
        pin = KML.Placemark(
            KML.name(str(count)),
            KML.description(file_name),
            KML.Point(
                KML.coordinates(str(long) + "," + str(lat))
            )
        )
        KML_fldrs[var_name].append(pin)
        
    for var_name, KML_fld in KML_fldrs.items():
        with open(os.path.join(write_folder, kml_pinmap_fn + '_' + var_name + '.kml'), 'w') as f_w:
            f_w.write(lxml.etree.tostring(KML_fld, pretty_print=True).decode())
            

    #Save KML pin map if we have all the variables ['VHM0', 'VAVH', 'WSPD']
    if all([v in variables for v in ['VHM0', 'VAVH', 'WSPD']]):
        #Number of  overlaps
        mult_var = result_df.groupby(['sar_url', 'bouy_file_name', 'bouy_longitude', 'bouy_latitude'])[['bouy_variable_name']].aggregate(lambda tdf: tdf.unique().tolist())
        mult_var['len'] = mult_var['bouy_variable_name'].apply(len)
        mult_var['contains_wind'] = mult_var['bouy_variable_name'].apply(lambda x: 'WSPD' in x)
        mult_var['also_VAVH'] = mult_var['bouy_variable_name'].apply(lambda x: 'VAVH' in x)
        mult_var['also_VHM0'] = mult_var['bouy_variable_name'].apply(lambda x: 'VHM0' in x)
        mult_var = mult_var[(mult_var['len'] > 1) & mult_var['contains_wind']].drop(labels='contains_wind', axis=1)

        wind_and_VAVH = mult_var[mult_var['also_VAVH']].drop(labels=['also_VAVH', 'also_VHM0'], axis=1).groupby(['bouy_file_name', 'bouy_longitude', 'bouy_latitude'])['len'].count()
        wind_and_VHM0 = mult_var[mult_var['also_VHM0']].drop(labels=['also_VAVH', 'also_VHM0'], axis=1).groupby(['bouy_file_name', 'bouy_longitude', 'bouy_latitude'])['len'].count()

        #Save overlapping wind and wave info to KML maps
        for kml_pinmap_fn, data_var in [('WSPD_and_VHM0', wind_and_VHM0), ('WSPD_and_VAVH', wind_and_VAVH)]:
            KML_fldr = KML.Folder(
                KML.name(kml_pinmap_fn + " " + write_folder[write_folder.find('/')+1:]),
                KML.description('Total datapoints: ' + str(data_var.sum())),
            )

            #iterate over all unique variable name, file name pairs
            for ((file_name, long, lat), count) in data_var.items():
                #Create the pin
                pin = KML.Placemark(
                    KML.name(str(count)),
                    KML.description(file_name),
                    KML.Point(
                        KML.coordinates(str(long) + "," + str(lat))
                    )
                )
                KML_fldr.append(pin)

            with open(os.path.join(write_folder, kml_pinmap_fn + '_' + var_name + '.kml'), 'w') as f_w:
                f_w.write(lxml.etree.tostring(KML_fldr, pretty_print=True).decode())


