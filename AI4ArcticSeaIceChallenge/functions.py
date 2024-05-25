#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helping functions for 'introduction' and 'quickstart' notebooks."""

# -- File info -- #
__author__ = 'Andreas R. Stokholm'
__contributors__ = ''
__copyright__ = ['Technical University of Denmark', 'European Space Agency']
__contact__ = ['stokholm@space.dtu.dk']
__version__ = '1.0.0'
__date__ = '2022-10-17'

# -- Built-in modules -- #

# -- Third-party modules -- #
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, f1_score

# -- Proprietary modules -- #
from AI4ArcticSeaIceChallenge.utils import ICE_STRINGS, GROUP_NAMES


def chart_cbar(ax, n_classes, chart, cmap='vridis'):
    """
    Create discrete colourbar for plot with the sea ice parameter class names.

    Parameters
    ----------
    n_classes: int
        Number of classes for the chart parameter.
    chart: str
        The relevant chart.
    """
    arranged = np.arange(0, n_classes)
    cmap = plt.get_cmap(cmap, n_classes - 1)
    norm = mpl.colors.BoundaryNorm(arranged - 0.5, cmap.N)  # Get colour boundaries. -0.5 to center ticks for each color.
    arranged = arranged[:-1]  # Discount the mask class.
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=arranged, fraction=0.0485, pad=0.049, ax=ax)
    cbar.set_label(label=ICE_STRINGS[chart])
    cbar.set_ticklabels(list(GROUP_NAMES[chart].values()))


def compute_metrics(true, pred, charts, metrics):
    """
    Calculates metrics for each chart and the combined score. true and pred must be 1d arrays of equal length. 

    Parameters
    ----------
    true : 
        ndarray, 1d contains all true pixels. Must be numpy array.
    pred :
        ndarray, 1d contains all predicted pixels. Must be numpy array.
    charts : List
        List of charts.
    metrics : Dict
        Stores metric calculation function and weight for each chart.

    Returns
    -------
    combined_score: float
        Combined weighted average score.
    scores: list
        List of scores for each chart.
    """
    scores = {}
    for chart in charts:
        if true[chart].ndim == 1 and pred[chart].ndim == 1:
            scores[chart] = np.round(metrics[chart]['func'](true=true[chart], pred=pred[chart]) * 100, 3)
        else:
            print(f"true and pred must be 1D numpy array, got {true['SIC'].ndim} and {pred['SIC'].ndim} dimensions with shape {true['SIC'].shape} and {pred.shape}, respectively")
    
    combined_score = compute_combined_score(scores=scores, charts=charts, metrics=metrics)

    return combined_score, scores


def r2_metric(true, pred):
    """
    Calculate the r2 metric.

    Parameters
    ----------
    true : 
        ndarray, 1d contains all true pixels. Must by numpy array.
    pred :
        ndarray, 1d contains all predicted pixels. Must by numpy array.

    Returns
    -------
    r2 : float
        The calculated r2 score.
        
    """
    r2 = r2_score(y_true=true, y_pred=pred)
    
    return r2
    
    
def f1_metric(true, pred):
    """
    Calculate the weighted f1 metric.

    Parameters
    ----------
    true : 
        ndarray, 1d contains all true pixels.
    pred :
        ndarray, 1d contains all predicted pixels.

    Returns
    -------
    f1 : float
        The calculated f1 score.
        
    """
    f1 = f1_score(y_true=true, y_pred=pred, average='weighted')
    
    return f1


def compute_combined_score(scores, charts, metrics):
    """
    Calculate the combined weighted score.

    Parameters
    ----------
    scores : List
        Score for each chart.
    charts : List
        List of charts.
    metrics : Dict
        Stores metric calculation function and weight for each chart.

    Returns
    -------
    : float
        The combined weighted score.
        
    """
    combined_metric = 0
    sum_weight = 0
    for chart in charts:
        combined_metric += scores[chart] * metrics[chart]['weight']
        sum_weight += metrics[chart]['weight']
    
    return np.round(combined_metric / sum_weight, 3)
