#!/usr/bin/env python3
"""
This library includes functions for plotting and calculating metrics to standarize the evaluation of the models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.metrics as sk_metrics

def sign(num):
    if num < 0:
        return '-'
    return '+'

"""
Plot a heatmap of the data

best_line Specifies if the best fitted line should be plotted or not
"""
def heatmap(df, name_text, unit, target_column, prediction_column, cmap='viridis', best_line=False):
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    max_value = max(df[target_column].max(), df[prediction_column].max())

    my_cmap = mpl.cm.get_cmap(cmap)
    (h2d_h, h2d_xedg, h2d_yedg, h2d_img) = ax.hist2d(df[target_column], df[prediction_column], bins=100, norm=mpl.colors.LogNorm(clip=True), cmap=my_cmap)
    ax.plot([0, 1], [0, 1], color='black', transform=ax.transAxes, label='Identity line x=y') #Plot identity line
    
    if best_line:
        slope, intersection = np.polyfit(df[target_column], df[prediction_column], deg=1)
        ax.plot([0, max_value], [intersection, intersection+max_value*slope], color='red', label=f'Best fitted line y={round(slope,2)}x {sign(intersection)} {round(abs(intersection),2)}')
    
    ax.set_title('Heatmap of ' + name_text + ', log scaled colormap')
    ax.set_xlabel(f"Target {name_text} [{unit}]")
    ax.set_ylabel(f"Predicted {name_text} [{unit}]")
    fig.colorbar(h2d_img, ax=ax, label='Count colormap')
    ax.set_facecolor(my_cmap(0))
    ax.legend()
    ax.set(xlim=(0, max_value), ylim=(0, max_value))

    plt.close()
    return fig

"""
Calculate the RMSE of the data
"""
def rmse(df, target_column, prediction_column):
    rmse = sk_metrics.mean_squared_error(df[target_column], df[prediction_column], squared=False)
    return rmse

"""
Calculate the bias of the data
"""
def bias(df, target_column, prediction_column):
    bias = np.mean(df[prediction_column] - df[target_column])
    return bias

"""
Calculate the slope of the data
"""
def slope(df, target_column, prediction_column):
    slope = np.polyfit(df[target_column], df[prediction_column], deg=1)[0]
    return slope
    
"""
Calculate all of the metrics of the data and return as a dictionary
"""
def metrics(df, target_column, prediction_column):
    rmse_metric = rmse(df, target_column, prediction_column)
    bias_metric = bias(df, target_column, prediction_column)
    slope_metric = slope(df, target_column, prediction_column)
    data_points = df.shape[0]
    return {'rmse':rmse_metric, 'bias':bias_metric, 'slope':slope_metric, 'data_points':data_points}