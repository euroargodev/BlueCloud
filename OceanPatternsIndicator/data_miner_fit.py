import sys
import xarray as xr
import numpy as np
import pyxpcm
from pyxpcm.models import pcm
import Plotter
from Plotter import Plotter
from BIC_calculation import *
import dask
import dask.array as da


def get_args():
    """
    Extract arguments from command line

    Returns
    -------
    parse.parse_args(): dict of the arguments

    """
    import argparse

    parse = argparse.ArgumentParser(description="Ocean patterns method")
    parse.add_argument('k', type=int, help="number of clusters K")
    parse.add_argument('file_name', type=str, help='input dataset')

    return parse.parse_args()


def load_data(file_name):
    """
    Load dataset into a Xarray dataset

    Parameters
    ----------
    file_name : Path to the NetCDF dataset

    Returns
    -------
    ds: Xarray dataset

    """
    ds = xr.open_dataset('../datasets/' + file_name)
    ds['depth'] = -np.abs(ds['depth'].values)
    ds.depth.attrs['axis'] = 'Z'
    return ds


def train_model(k, ds, var_name_mdl, var_name_ds, z_dim):
    """
    Train pyXpcm model and predict values of dataset

    Parameters
    ----------
    k : number of clusters
    ds : Xarray dataset
    var_name_mdl : name of variable in model
    var_name_ds : name of variable in dataset
    z_dim : z axis dimension (depth)

    Returns
    -------
    m: Trained model
    ds: Xarray dataset with predicted values
    """
    # create model
    z = ds[z_dim][0:30]
    pcm_features = {var_name_mdl: z}
    m = pcm(K=k, features=pcm_features)
    # fit model
    features_in_ds = {var_name_mdl: var_name_ds}
    m.fit_predict(ds, features_in_ds, dim=z_dim, inplace=True)
    return m, ds


def robustness(m, ds, features_in_ds, z_dim):
    """
    Compute the robustness of the trained model: The PCM robustness represents a usefull scaled probability of a profile
    to belong to a class. If a lot of profiles show very low values you should maybe change the number of classes.
    Parameters
    ----------
    m : trained model
    ds : Xarray Dataset
    features_in_ds : dict {var_name_mdl: var_name_ds} with var_name_mdl the name of the variable in the model and
    var_name_ds the name of the variable in the dataset
    z_dim : z axis dimension (depth)
    Returns
    -------
    Saves the robustness plot to png
    """
    p = Plotter(ds, m)
    m.predict_proba(ds, features=features_in_ds, dim=z_dim, inplace=True)
    ds.pyxpcm.robustness(m, inplace=True)
    ds.pyxpcm.robustness_digit(m, inplace=True)
    p.plot_robustness(time_slice="2018-01-01")
    p.save_BlueCloud('dataminer_out/robustness.png', bic_fig='yes')


def main():
    z_dim = 'depth'
    var_name_ds = 'thetao'  # name in dataset
    var_name_mdl = 'temperature'  # name in model
    features_in_ds = {var_name_mdl: var_name_ds}
    args = get_args()
    k = args.k
    file_name = args.file_name
    ds = load_data(file_name)
    m, ds = train_model(k=k, ds=ds, var_name_mdl=var_name_mdl, var_name_ds=var_name_ds, z_dim=z_dim)
    robustness(m=m, ds=ds, features_in_ds=features_in_ds, z_dim=z_dim)
    m.to_netcdf('dataminer_out/model.nc')


if __name__ == '__main__':
    main()
