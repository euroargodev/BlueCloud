import logging

from utils.preprocessing_OR import OR_unstack_dataset
from utils.Plotter_OR import Plotter_OR
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def generate_dev_plots(ds, model, var_name_ds, ds_init, mask):
    """
    Robustness: Robustness is a scaled probability of a time series to belong to a class. When looking at the spatial
    distribution of the robustness metric, and if classes have a spatial structure, you may encounter regions with high
    probabilities: these regions are the "core" of the class.

    Parameters
    ----------
    mask : mask used to delete NaNs, used for unstack
    ds_init : Initial dataset to keep all the attributes
    model : trained model
    ds : Xarray dataset containing the predictions
    var_name_ds : name of the variable in the dataset
    Returns
    -------
    saves all the plots as png
    """
    x_proba = model.predict_proba(ds[var_name_ds + "_reduced"])
    ds = ds.assign(variables={"GMM_post": (('sampling', 'k'), x_proba)})
    # calculate robustness
    maxpost = ds["GMM_post"].max(dim="k")
    nk = len(ds["GMM_labels"])
    robust = (maxpost - 1. / nk) * nk / (nk - 1.)
    plist = [0, 0.33, 0.66, 0.9, .99, 1]
    rowl0 = ('Unlikely', 'As likely as not', 'Likely', 'Very Likely', 'Virtually certain')
    robust_id = np.digitize(robust, plist) - 1
    ds = ds.assign(variables={"GMM_robustness": ('sampling', robust), "GMM_robustness_cat": ('sampling', robust_id)})
    ds["GMM_robustness_cat"].attrs['legend'] = rowl0
    ds = OR_unstack_dataset(ds_init, ds, mask)

    P = Plotter_OR(ds, model)
    P.scatter_PDF(var_name=var_name_ds + '_reduced')
    P.save_BlueCloud('scatter_PDF.png')
    # robustness
    P.plot_robustness()
    P.save_BlueCloud('robustness.png')


def save_empty_plot(name):
    text = "This figure is not available,\n please refer to the logs to understand why."
    x = np.arange(0, 8, 0.1)
    y = np.sin(x)
    plt.plot(x, y)
    plt.figtext(0.5, 0.5,
                text,
                horizontalalignment="center",
                verticalalignment="center",
                wrap=True, fontsize=14,
                color="red",
                bbox={'facecolor': 'grey',
                      'alpha': 0.8, 'pad': 5})
    plt.savefig(f"{name}.png")
    return 0


def generate_plots(model, ds, var_name_ds):
    """
    Generates and saves the following plots:
    - Time series structure: The graphic representation of quantile time series reveals the seasonal structure of each
    class. The median time series will give you the best idea of the typical time series of a class and the other
    quantiles, the possible spread of time series within a class. You can choose the start month in the plot.
    - Time series per quantile
    - Spacial distribution of classes: plot the GMM labels in a map to analyse the spatial coherence of classes.
    The spatial information (coordinates) of time series is not used to fit the model, so spatial coherence appears
    naturally, revealing seasonal structure similarities between different areas of the ocean.
    - Robustness: Robustness is a scaled probability of a time series to belong to a class. When looking at the spatial
    distribution of the robustness metric, and if classes have a spatial structure, you may encounter regions with high
    probabilities: these regions are the "core" of the class.
    - Class pie chart: pie chart showing the percentage of profiles belonging to each class and the number of classified
     profiles.


    Parameters
    ----------
    model : trained model
    ds : Xarray dataset containing the predictions
    var_name_ds : name of the variable in the dataset
    Returns
    -------
    saves all the plots as png
    """
    try:
        y_label = ds[var_name_ds].attrs['long_name'] + " in " + ds[var_name_ds].attrs['unit_long']
    except KeyError:
        y_label = var_name_ds
    P = Plotter_OR(ds, model)

    # plot time series by class
    P.tseries_structure(q_variable=var_name_ds + '_Q', ylabel=y_label)
    P.save_BlueCloud('tseries_struc.png')
    # plot time series by quantile
    P.tseries_structure_comp(q_variable=var_name_ds + '_Q', plot_q='all', ylabel=y_label)
    P.save_BlueCloud('tseries_struc_comp.png')
    # spacial distribution
    P.spatial_distribution()
    P.save_BlueCloud('spatial_dist.png')
    # robustness
    P.plot_robustness()
    P.save_BlueCloud('robustness.png')
    # pie chart of the classes distribution
    P.pie_classes()
    P.save_BlueCloud('pie_chart.png')
    # save dataset predicted
    ds.to_netcdf('predicted_dataset.nc', format='NETCDF4')
    logging.info('saving predicted dataset in predicted_dataset.nc')


def predict(ds, var_name_ds, model):
    x_labels = model.predict(ds[var_name_ds + "_reduced"])
    ds = ds.assign(variables={"GMM_labels": ('sampling', x_labels)})
    return ds


def robustness(model, ds, var_name_ds):
    x_proba = model.predict_proba(ds[var_name_ds + "_reduced"])
    ds = ds.assign(variables={"GMM_post": (('sampling', 'k'), x_proba)})
    maxpost = ds["GMM_post"].max(dim="k")
    nk = len(ds["GMM_labels"])
    robust = (maxpost - 1. / nk) * nk / (nk - 1.)
    plist = [0, 0.33, 0.66, 0.9, .99, 1]
    rowl0 = ('Unlikely', 'As likely as not', 'Likely', 'Very Likely', 'Virtually certain')
    robust_id = np.digitize(robust, plist) - 1
    ds = ds.assign(variables={"GMM_robustness": ('sampling', robust), "GMM_robustness_cat": ('sampling', robust_id)})
    ds["GMM_robustness_cat"].attrs['legend'] = rowl0
    return ds


def quantiles(ds, var_name_ds, k, ds_init, mask):
    q = [0.05, 0.5, 0.95]
    k_values = np.unique(ds['GMM_labels'].values)
    nan_matrix = np.empty((k, np.size(q), np.size(ds.feature)))
    nan_matrix[:] = np.NaN
    m_quantiles = xr.DataArray(nan_matrix, dims=['k', 'quantile', 'feature'])
    for yi in range(k):
        if yi in k_values:
            m_quantiles[yi] = ds[var_name_ds].where(ds['GMM_labels'] == yi, drop=True).quantile(q, dim='sampling')
    ds = ds.assign(variables={var_name_ds + "_Q": (('k', 'quantile', 'feature'), m_quantiles)})
    ds = ds.assign_coords(coords={'quantile': q})

    # Unstack dataset
    ds_labels = OR_unstack_dataset(ds_init, ds, mask)
    return ds_labels
