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
    parse.add_argument('file_name', type=str, help='input dataset')
    parse.add_argument('nk', type=int, help='number max of clusters')

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


def bic_calculation(ds, features_in_ds, z_dim, var_name_mdl, nk):
    """
    The BIC (Bayesian Information Criteria) can be used to optimize the number of classes in the model, trying not to
    over-fit or under-fit the data. To compute this index, the model is fitted to the training dataset for a range of K
     values from 0 to 20. A minimum in the BIC curve will give you the optimal number of classes to be used.
    Parameters
    ----------
    ds : Xarray dataset
    features_in_ds : dict {var_name_mdl: var_name_ds} with var_name_mdl the name of the variable in the model and
    var_name_ds the name of the variable in the dataset
    z_dim : z axis dimension (depth)
    var_name_mdl : name of the variable in the model
    nk : number of K to explore (always starts at 1 up to nk)

    Returns
    -------
    Saves the BIC plot in png with the min of the BIC indicated on it
    """
    z = ds[z_dim][0:30]
    pcm_features = {var_name_mdl: z}
    corr_dist = 50  # correlation distance in km
    time_steps = ['2018-01', '2018-07']  # time steps to be used into account
    nrun = 10  # number of runs for each k
    bic, bic_min = BIC_calculation(ds=ds, coords_dict=get_coords_dict(ds),
                                   corr_dist=corr_dist, time_steps=time_steps,
                                   pcm_features=pcm_features, features_in_ds=features_in_ds, z_dim=z_dim,
                                   Nrun=nrun, NK=nk)
    plot_BIC(BIC=bic, NK=nk, bic_min=bic_min)
    plt.savefig('dataminer_out/BIC.png', bbox_inches='tight', pad_inches=0.1)


def main():
    z_dim = 'depth'
    var_name_ds = 'thetao'  # name in dataset
    var_name_mdl = 'temperature'  # name in model
    features_in_ds = {var_name_mdl: var_name_ds}
    args = get_args()
    file_name = args.file_name
    nk = args.nk
    ds = load_data(file_name)
    bic_calculation(ds=ds, features_in_ds=features_in_ds, z_dim=z_dim, var_name_mdl=var_name_mdl, nk=nk)


if __name__ == '__main__':
    main()
