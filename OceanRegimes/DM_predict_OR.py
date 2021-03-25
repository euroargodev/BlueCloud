import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import mixture
from preprocessing_OR import *
import Plotter_OR
from Plotter_OR import Plotter_OR
import joblib
import time


def get_args():
    """
    Extract arguments from command line

    Returns
    -------
    parse.parse_args(): dict of the arguments

    """
    import argparse

    parse = argparse.ArgumentParser(description="Ocean patterns method")
    parse.add_argument('model', type=str, help="input model")
    parse.add_argument('file_name', type=str, help='input dataset')
    parse.add_argument('mask', type=str, help='path to mask')
    parse.add_argument('var_name_ds', type=str, help='name of variable in dataset')
    return parse.parse_args()


def load_data(file_name, var_name_ds):
    """
    Load dataset into a Xarray dataset

    Parameters
    ----------
    var_name_ds : name of variable in dataset
    file_name : Path to the NetCDF dataset

    Returns
    -------
    ds: Xarray dataset
    """
    ds = xr.open_dataset(file_name)
    # select var
    ds = ds[[var_name_ds]]
    # some format
    ds['time'] = ds.indexes['time'].to_datetimeindex()
    ds.time.attrs['axis'] = 'T'
    return ds


def preprocessing_ds(ds, var_name_ds, mask_path):
    """
    5 steps of the preprocessing, detailed code in the preprocessing_OR.py script:
    - Weekly mean
    - Reduce latitude and longitude to sampling dim
    - Delete all NaN values (using a mask that can be given as an input)
    - Scaler: default is scikit-learn StandardScaler
    - Principal Component Analysis (PCA): n_components default value is 0.99
    Parameters
    ----------
    ds : input dataset (Xarray)
    var_name_ds : name of variable in dataset
    mask_path : path to mask, default is auto and the mask will be generated automatically

    Returns
    -------

    """
    x = OR_weekly_mean(ds=ds, var_name=var_name_ds)
    x = OR_reduce_dims(X=x)
    try:
        x, mask = OR_delate_NaNs(X=x, var_name=var_name_ds, mask_path=mask_path)
    except FileNotFoundError as e:
        print("no mask was found, generating one: " + str(e.filename))
        x, mask = OR_delate_NaNs(X=x, var_name=var_name_ds, mask_path='auto')
    x = OR_scaler(X=x, var_name=var_name_ds)
    x = OR_apply_PCA(X=x, var_name=var_name_ds)
    return x, mask


def load_model(model_path):
    """
    Load trained model
    Parameters
    ----------
    model_path : path of model to load

    Returns
    -------
    model: trained sklearn GMM model
    k: number of class
    """
    model = joblib.load(model_path)
    k = model.n_components
    return model, k


def predict(model, ds, var_name_ds, k, ds_init, mask):
    """
    - Predict classes from a dataset with a trained GMM model
    - Compute robustness
    - Compute quantiles
    Parameters
    ----------
    mask : mask used to delete NaNs, used for unstack
    ds_init : Initial dataset to keep all the attributes
    k : number of class
    model : Trained model
    ds : Xarray dataset
    var_name_ds : name of variable in dataset

    Returns
    -------
    ds: Xarray dataset with the predicted classification for each profile
    """
    # predict class
    x_labels = model.predict(ds[var_name_ds + "_reduced"])
    ds = ds.assign(variables={"GMM_labels": ('sampling', x_labels)})

    # compute robustness
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

    # compute quantiles
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
    P.tseries_structure(q_variable=var_name_ds + '_Q', start_month=6, ylabel=y_label)
    P.save_BlueCloud('tseries_struc.png')
    # plot time series by quantile
    P.tseries_structure_comp(q_variable=var_name_ds + '_Q', plot_q='all', ylabel=y_label, start_month=6)
    P.save_BlueCloud('tseries_struc_comp.png')
    # spacial distribution
    P.spatial_distribution()
    P.save_BlueCloud('spatial_distr.png')
    # robustness
    P.plot_robustness()
    P.save_BlueCloud('robustness.png')
    # pie chart of the classes distribution
    P.pie_classes()
    P.save_BlueCloud('pie_chart.png')
    # save dataset predicted
    ds.to_netcdf('predicted_dataset.nc', format='NETCDF4')


def main():
    args = get_args()
    var_name_ds = args.var_name_ds
    file_name = args.file_name
    mask_path = args.mask
    model_path = args.model

    print("loading the dataset")
    start_time = time.time()
    ds_init = load_data(file_name=file_name, var_name_ds=var_name_ds)
    load_time = time.time() - start_time
    print("load finished in " + str(load_time) + "sec")

    print("preprocess the dataset")
    start_time = time.time()
    ds, mask = preprocessing_ds(ds=ds_init, var_name_ds=var_name_ds, mask_path=mask_path)
    load_time = time.time() - start_time
    print("preprocessing finished in " + str(load_time) + "sec")

    print("starting computation")
    start_time = time.time()
    model, k = load_model(model_path=model_path)
    train_time = time.time() - start_time
    print("training finished in " + str(train_time) + "sec")

    start_time = time.time()
    ds = predict(model=model, ds=ds, var_name_ds=var_name_ds, k=k, mask=mask, ds_init=ds_init)
    predict_time = time.time() - start_time
    print("prediction finished in " + str(predict_time) + "sec")

    start_time = time.time()
    generate_plots(model=model, ds=ds, var_name_ds=var_name_ds)
    plot_time = time.time() - start_time
    print("plots finished in " + str(plot_time) + "sec")


if __name__ == '__main__':
    main()
