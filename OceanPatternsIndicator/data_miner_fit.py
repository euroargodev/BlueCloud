import sys
import xarray as xr
import numpy as np
import pyxpcm
from pyxpcm.models import pcm
import Plotter
from Plotter import Plotter
import dask
import dask.array as da
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
    parse.add_argument('k', type=int, help="number of clusters K")
    parse.add_argument('file_name', type=str, help='input dataset')
    parse.add_argument('var_name_ds', type=str, help='name of variable in dataset')
    parse.add_argument('var_name_mdl', type=str, help='name of variable in model')

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
    first_date: string, first time slice of the dataset
    coord_dict: coordinate dictionary for pyXpcm
    """
    ds = xr.open_dataset(file_name)
    # select var
    ds = ds[var_name_ds].to_dataset()
    first_date = str(ds.time.min().values)[0:7]
    # exception to handle missing depth dim: setting depth to 0 because the dataset most likely represents surface data
    try:
        coord_dict = get_coords_dict(ds)
        ds['depth'] = -np.abs(ds[coord_dict['depth']].values)
        ds.depth.attrs['axis'] = 'Z'
    except KeyError as e:
        ds = ds.expand_dims('depth').assign_coords(depth=("depth", [0]))
        ds.depth.attrs['axis'] = 'Z'
        coord_dict = get_coords_dict(ds)
        print(f"{e} dimension was missing,it has been initialized to 0 for surface data")
    return ds, first_date, coord_dict


def get_coords_dict(ds):
    """
    create a dict of coordinates to mapping each dimension of the dataset
    Parameters
    ----------
    ds : Xarray dataset

    Returns
    -------
    coords_dict: dict mapping each dimension of the dataset
    """
    # creates dictionary with coordinates
    coords_list = list(ds.coords.keys())
    coords_dict = {}
    for c in coords_list:
        axis_at = ds[c].attrs.get('axis')
        if axis_at == 'Y':
            coords_dict.update({'latitude': c})
        if axis_at == 'X':
            coords_dict.update({'longitude': c})
        if axis_at == 'T':
            coords_dict.update({'time': c})
        if axis_at == 'Z':
            coords_dict.update({'depth': c})
    return coords_dict


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
    z = ds[z_dim]
    pcm_features = {var_name_mdl: z}
    m = pcm(K=k, features=pcm_features)
    # fit model
    features_in_ds = {var_name_mdl: var_name_ds}
    m.fit_predict(ds, features_in_ds, dim=z_dim, inplace=True)
    return m, ds


def robustness(m, ds, features_in_ds, z_dim, first_date):
    """
    Compute the robustness of the trained model: The PCM robustness represents a useful scaled probability of a profile
    to belong to a class. If a lot of profiles show very low values you should maybe change the number of classes.
    Parameters
    ----------
    m : trained model
    ds : Xarray Dataset
    features_in_ds : dict {var_name_mdl: var_name_ds} with var_name_mdl the name of the variable in the model and
    var_name_ds the name of the variable in the dataset
    z_dim : z axis dimension (depth)
    first_date : first time slice from dataset
    Returns
    -------
    Saves the robustness plot to png
    """
    p = Plotter(ds, m)
    m.predict_proba(ds, features=features_in_ds, dim=z_dim, inplace=True)
    ds.pyxpcm.robustness(m, inplace=True)
    ds.pyxpcm.robustness_digit(m, inplace=True)
    p.plot_robustness(time_slice=first_date)
    p.save_BlueCloud('robustness.png', bic_fig='yes')


def main():
    args = get_args()
    var_name_ds = args.var_name_ds
    var_name_mdl = args.var_name_mdl
    features_in_ds = {var_name_mdl: var_name_ds}
    k = args.k
    file_name = args.file_name
    print("loading the dataset")
    start_time = time.time()
    ds, first_date, coord_dict = load_data(file_name)
    z_dim = coord_dict['depth']
    load_time = time.time() - start_time
    print("load finished in " + str(load_time) + "sec")
    print("starting computation")
    start_time = time.time()
    m, ds = train_model(k=k, ds=ds, var_name_mdl=var_name_mdl, var_name_ds=var_name_ds, z_dim=z_dim)
    train_time = time.time() - start_time
    print("computation finished in " + str(train_time) + "sec")
    robustness(m=m, ds=ds, features_in_ds=features_in_ds, z_dim=z_dim, first_date=first_date)
    print("robustness computation finished, plot saved")
    m.to_netcdf('model.nc')
    print("model saved")


if __name__ == '__main__':
    main()
