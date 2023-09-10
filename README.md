# Predicting sea surface wave and wind parameters from satellite radar images using machine learning

## Project Overview

Accurate predictions of wave and wind parameters over oceans are crucial for various marine operations. Although buoys provide accurate measurements, their deployment is limited, which necessitates the exploration of alternative data sources. Sentinel-1, a satellite mission capturing Synthetic Aperture Radar (SAR) images with high coverage, presents a promising opportunity. This project aims to find the relationship between SAR images and wave/wind parameters to make accurate predictions. The methodology encompasses training deep learning models using features and sub-images extracted from SAR images and correlating them with buoy measurements and model data.

## Getting Started

To utilize this project, the user needs to have access to the following data:
- **Buoy Data**: Downloaded for the year 2021 from [DOI 10.48670/moi-00036](https://doi.org/10.48670/moi-00036)
- **Model Data**:
  - Significant Wave Height: [DOI 10.24381/cds.adbb2d47](https://doi.org/10.24381/cds.adbb2d47)
  - Wind Speed: [DOI 10.48670/moi-00185](https://doi.org/10.48670/moi-00185)
- **SAR Data**: Obtainable from the Alaska Satellite Facility through the [ASF SeachAPI](https://search.asf.alaska.edu/#/)

## Repository Structure

The repository holds multiple files dedicated to analysis, plotting, experimentation, and other project-related tasks. However, the key files and directories needed to run the project from raw data to predictions are detailed below:

### Buoy Survey

- `survey.py`
  - **Description**: Code to perform collocation, identifying all SAR images covering locations found in the buoy data and labels them using the closest measurements. Also creates .kml pinmaps for visualizing the buoy data.
  - **Key Input**: Downloaded buoy data.
  - **Key Output**: Dataframe relating buoy measurements to the corresponding SAR image.

### Pipeline

- `download_sar_multiprocess.py`
  - **Description**: Utilizes the dataframe from `bouy_survey/survey.py` to extract sub-images around each measurement, storing each as a NetCDF file.
  - **Key Input**: Dataframe from `bouy_survey/survey.py`.
  - **Key Output**: Sub-images from all measurements, stored as NetCDF files.

- `analyze_hom_filter_dataset.ipynb`
  - **Description**: Demonstrates the creation and functionality of the homogeneity filter. Allows users to utilize the existing filter or supply their own.
  - **Key Input**: Dataset of homogenous and non homogenous images.
  - **Key Output**: Support Vector Classifier (SVC) that differentiates the classes.
  
- `sar_dataset_loader.py`
  - **Description**: Details the process of compiling the final dataset, including feature calculation from sub-images and setting up the data split for model training.
  - **Key Input**: Dataframe from `bouy_survey/survey.py`, sub-image directory from `download_sar_multiprocess.py`, homogeneity filter SVC from `analyze_hom_filter_dataset.ipynb`.
  - **Key Output**: Dataframe relating features and labels. 
  
- `data_copy.py`
  - **Description**: Enables copying datasets to a new location (mostly intended for moving data to Alvis node) according to user specifications, for example regarding swath, polarization, size, nan-values, train-test-val split and homogeneity. 
  - **Key Input**: Sub-image directory from `download_sar_multiprocess.py`, dataframe from `sar_dataset_loader.py`.
  - **Key Output**: Dataset according to specifications in new location.
  
### Machine Learning

`machine_learning` holds scripts for training, analyzing, and testing the deep learning models. 

#### Library File:

- `sarssw_ml_lib.py`
  - **Description**: Repository of machine learning functions, including the loss function, PyTorch network architecture, and data loaders.

#### Script Breakdown:

The scripting files are organized using a naming convention which describes their role in the project:

- Files prefixed with `run_` are shell scripts designed to execute the corresponding Python scripts on the Alvis cluster.
- Files containing `feature_nn` relate to the deep learning model that only uses the extracted features. Conversely, those containing `image_feature_nn` involve a model that also employs a CNN as a feature extractor.
- The files suffixed with:
  - `_optuna.py` are used for hyperparameter tuning of the respective model.
  - `_train.py` trains a specific model with determined hyperparameters, saving the best ones.
  - `_test.py` tests the trained models, concluding the process from raw data to parameter predictions.

## Precomputed data

#### This repository
In this repository, the following files are of special interest for running the project. A pickle file of the data frame after the collocation in `bouy_survey/survey.py` can be found in `bouy_survey/1h_survey/result_df`. A precomputed SVC homogeneity filter can be found in `sar_survey/out/homogenity_svc.pkl`

#### Quartz server
For those with access to the quartz server for this project, the dataset of sub-images from `pipeline/download_sar_multiprocess.py` can be found in `/data/exjobb/sarssw/sar_dataset`. The most recent dataset of features and lables can be found in `/data/exjobb/sarssw/sar_dataset_features_lables_22_may/sar_dataset.pickle`. The predictions of the final models can be found in  `/data/exjobb/sarssw/result_predictions` and `result_predictions_only_features`. 

#### Mimer server
For those with access to the mimer server for this project, the sub-image dataset, the features and lables dataset and the final predictions can be found with the same names in `/mimer/NOBACKUP/priv/chair/sarssw/`. In this directory, a few datasets extracted using `pipeline/data_copy.py` according to different specifications can be found. Most noteably, `IW_VV_VH` is the full dataset according to the limitations of this project, but other versions can easily be recomputed with `pipeline/data_copy.py`. Finally, the checkpoints of the final models can be found in `final_img_feat/version_0/checkpoints/` and `final_only_feat/version_1/checkpoints/`
