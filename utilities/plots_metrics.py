#!/usr/bin/env python3
"""
This library includes functions for plotting and calculating metrics to standarize the evaluation of the models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import sklearn.metrics as sk_metrics

def sign(num):
    if num < 0:
        return '-'
    return '+'

def heatmap(df, name_text, title, unit, target_column, prediction_column, best_line=False, target_label_override=None, prediction_label_override=None, save_as=None):
    """
    Plot a heatmap of the data

    best_line Specifies if the best fitted line should be plotted or not
    """
    colors = {
        "cmap": "cividis",         # Heatmap color scheme
        "identity_line": "white", # Identity line color
        "best_fit_line": "red",   # Best fit line color
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    max_value = max(df[target_column].max(), df[prediction_column].max())

    (h2d_h, h2d_xedg, h2d_yedg, h2d_img) = ax.hist2d(df[target_column], df[prediction_column], bins=100, norm=mpl.colors.LogNorm(clip=True), cmap=colors["cmap"])
    ax.plot([0, 1], [0, 1], color=colors["identity_line"], transform=ax.transAxes, label='Identity line x=y') #Plot identity line
    
    if best_line:
        slope, intersection = np.polyfit(df[target_column], df[prediction_column], deg=1)
        ax.plot([0, max_value], [intersection, intersection+max_value*slope], color=colors["best_fit_line"], label=f'Best fitted line y={round(slope,2)}x {sign(intersection)} {round(abs(intersection),2)}')
    
    ax.set_title('Heatmap of ' + title)
    if target_label_override is not None:
        ax.set_xlabel(target_label_override)
    else:  
        ax.set_xlabel(f"Target {name_text} [{unit}]")
        
    if prediction_label_override is not None:
        ax.set_ylabel(prediction_label_override)
    else:
        ax.set_ylabel(f"Predicted {name_text} [{unit}]")
        
    fig.colorbar(h2d_img, ax=ax, label='Count colormap')
    ax.set_facecolor(plt.get_cmap(colors["cmap"])(0))
    
    metrics_dict = metrics(df, target_column, prediction_column, include_data_points=False)
    metric_string = [n+": "+str(round(v,3)) for n,v in metrics_dict.items()]
    metric_string = '\n'.join(metric_string)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label=metric_string))
    ax.legend(handles=handles)

    ax.set(xlim=(0, max_value), ylim=(0, max_value))
    plt.close()
    
    if save_as is not None:
        fig.savefig(save_as, dpi=300, bbox_inches='tight')
    return fig

def my_histogram(data, title, xlabel, ylabel="Count", bins=100):  
    """
    Plot a histogram of the data
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(data, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    plt.close()
    return fig
    
def rmse(df, target_column, prediction_column):
    """
    Calculate the RMSE of the data
    """
    rmse = sk_metrics.mean_squared_error(df[target_column], df[prediction_column], squared=False)
    return rmse

def bias(df, target_column, prediction_column):
    """
    Calculate the bias of the data
    """
    bias = np.mean(df[prediction_column] - df[target_column])
    return bias

def slope(df, target_column, prediction_column):
    """
    Calculate the slope of the data
    """
    slope = np.polyfit(df[target_column], df[prediction_column], deg=1)[0]
    return slope

def correlation(df, target_column, prediction_column):
    """
    Calculate the correlation of the data
    """
    correlation = np.corrcoef(df[target_column], df[prediction_column])[0][1]
    return correlation

def metrics(df, target_column, prediction_column, include_data_points=True):
    """
    Calculate all of the metrics of the data and return as a dictionary
    """
    rmse_metric = rmse(df, target_column, prediction_column)
    bias_metric = bias(df, target_column, prediction_column)
    slope_metric = slope(df, target_column, prediction_column)
    correlation_metric = correlation(df, target_column, prediction_column)
    data_points = df.shape[0]
    if include_data_points:
        return {'RMSE':rmse_metric, 'Bias':bias_metric, 'Slope':slope_metric, 'Correlation':correlation_metric, 'Data points':data_points}

    return {'RMSE':rmse_metric, 'Bias':bias_metric, 'Slope':slope_metric, 'Correlation':correlation_metric}