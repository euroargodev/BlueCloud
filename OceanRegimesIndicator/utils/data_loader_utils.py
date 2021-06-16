import numpy as np
import xarray as xr
import logging
from utils.preprocessing_OR import *


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
    logging.info(f"dataset to load: {file_name}")
    ds = xr.open_mfdataset(file_name).load()
    # select var
    ds = ds[[var_name_ds]]
    # some format
    if not np.issubdtype(ds.indexes['time'].dtype, np.datetime64):
        logging.info("casting time to datetimeindex")
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
        logging.exception("no mask was found, generating one: " + str(e.filename))
        x, mask = OR_delate_NaNs(X=x, var_name=var_name_ds, mask_path='auto')
    x = OR_scaler(X=x, var_name=var_name_ds)
    x = OR_apply_PCA(X=x, var_name=var_name_ds)
    return x, mask