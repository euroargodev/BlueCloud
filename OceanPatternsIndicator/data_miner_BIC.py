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
    parse.add_argument('file_name', type=str, help='input dataset')
    parse.add_argument('nk', type=int, help='number max of clusters')
    parse.add_argument('var_name_ds', type=str, help='name of variable in dataset')
    parse.add_argument('var_name_mdl', type=str, help='name of variable in model')
    parse.add_argument('corr_dist', type=int, help='correlation distance used for BIC')
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


def bic_calculation(ds, features_in_ds, z_dim, var_name_mdl, nk, corr_dist, coord_dict, first_date):
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
    bic: all values for the bic graph
    bic_min: min value of the bic
    """
    z = ds[z_dim]
    pcm_features = {var_name_mdl: z}
    # TODO choose one time frame if short or choose one winter/summer pair
    time_steps = [first_date]
    # time_steps = ['2018-01', '2018-07']  # time steps to be used into account
    nrun = 10  # number of runs for each k
    bic, bic_min = BIC_calculation(ds=ds, coords_dict=coord_dict,
                                   corr_dist=corr_dist, time_steps=time_steps,
                                   pcm_features=pcm_features, features_in_ds=features_in_ds, z_dim=z_dim,
                                   Nrun=nrun, NK=nk)
    return bic, bic_min


def main():
    args = get_args()
    file_name = args.file_name
    nk = args.nk
    var_name_ds = args.var_name_ds
    var_name_mdl = args.var_name_mdl
    corr_dist = args.corr_dist
    features_in_ds = {var_name_mdl: var_name_ds}
    print("loading the dataset")
    start_time = time.time()
    ds, first_date, coord_dict = load_data(file_name)
    z_dim = coord_dict['depth']
    load_time = time.time() - start_time
    print("load finished in " + str(load_time) + "sec")
    print("starting computation")
    start_time = time.time()
    bic, bic_min = bic_calculation(ds=ds, features_in_ds=features_in_ds, z_dim=z_dim, var_name_mdl=var_name_mdl, nk=nk,
                                   corr_dist=corr_dist, coord_dict=coord_dict, first_date=first_date)
    bic_time = time.time() - start_time
    print("computation finished in " + str(bic_time) + "sec")
    plot_BIC(BIC=bic, NK=nk, bic_min=bic_min)
    print("computation finished, saving png")
    plt.savefig('bic.png', bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    main()
