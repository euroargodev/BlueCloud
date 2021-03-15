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
    parse.add_argument('model', type=str, help="input model")
    parse.add_argument('file_name', type=str, help='input dataset')
    parse.add_argument('var_name_ds', type=str, help='name of variable in dataset')
    parse.add_argument('var_name_mdl', type=str, help='name of variable in model')
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
    first_date: string, first time slice of the dataset
    coord_dict: coordinate dictionary for pyXpcm
    """
    ds = xr.open_dataset(file_name)
    first_date = str(ds.time.min().values)[0:7]
    coord_dict = get_coords_dict(ds)
    ds['depth'] = -np.abs(ds[coord_dict['depth']].values)
    ds.depth.attrs['axis'] = 'Z'
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


def load_model(model_path):
    m = pyxpcm.load_netcdf(model_path)
    return m


def predict(m, ds, var_name_mdl, var_name_ds, z_dim):
    """
    Predict classes from a dataset
    Parameters
    ----------
    m : Trained model
    ds : Xarray dataset
    var_name_mdl : name of variable in model
    var_name_ds : name of variable in dataset
    z_dim : z axis dimension (depth)

    Returns
    -------
    ds: Xarray dataset with the predicted classification for each profile
    """
    features_in_ds = {var_name_mdl: var_name_ds}
    m.predict(ds, features=features_in_ds, dim=z_dim, inplace=True)
    m.predict_proba(ds, features=features_in_ds, dim=z_dim, inplace=True)
    ds = ds.pyxpcm.quantile(m, q=[0.05, 0.5, 0.95], of=var_name_ds, outname=var_name_ds + '_Q', keep_attrs=True,
                            inplace=True)
    ds.pyxpcm.robustness(m, inplace=True)
    ds.pyxpcm.robustness_digit(m, inplace=True)
    return ds


def generate_plots(m, ds, var_name_ds, first_date):
    """
    Generates and saves the following plots:
    - vertical structure: vertical structure of each classes. It draws the mean profile and the 0.05 and 0.95 quantiles
    - vertical structure comp: vertical structure graph but Quantiles are being plotted together to highlight
    differences between classes.
    - Spacial distribution: plot the PCM labels in a map to analyse the spatial coherence of classes.
    - Robustness: spacial distribution of a scaled probability of a profile to belong to a class.
    - Pie chart: pie chart showing the percentage of profiles belonging to each class and the number of
    classified profiles.
    - Temporal distribution by month: The bar plots represents the percentage of profiles in each class by month.
    - Temporal distribution by season: The bar plots represents the percentage of profiles in each class by season.
    Parameters
    ----------
    m : trained model
    ds : Xarray dataset containing the predictions
    var_name_ds : name of the variable in the dataset
    first_date: date of first time slice
    Returns
    -------
    saves all the plots as png
    """
    if ds[var_name_ds].attrs['unit_long'] and ds[var_name_ds].attrs['long_name']:
        x_label = ds[var_name_ds].attrs['long_name'] + " in " + ds[var_name_ds].attrs['unit_long']
    else:
        x_label = var_name_ds
    P = Plotter(ds, m)
    # plot profiles by class
    P.vertical_structure(q_variable=var_name_ds + '_Q', sharey=True, xlabel=x_label)
    P.save_BlueCloud('vertical_struc.png')
    # plot profiles by quantile
    P.vertical_structure_comp(q_variable=var_name_ds + '_Q', plot_q='all', xlabel=x_label, ylim=[-1000, 0])
    P.save_BlueCloud('vertical_struc_comp.png')
    # spacial distribution
    P.spatial_distribution(time_slice='most_freq_label')
    P.save_BlueCloud('spatial_distr_freq.png')
    # robustness
    P.plot_robustness(time_slice=first_date)
    P.save_BlueCloud('robustness.png')
    # pie chart of the classes distribution
    P.pie_classes()
    P.save_BlueCloud('pie_chart.png')
    # temporal distribution (monthly)
    P.temporal_distribution(time_bins='month')
    P.save_BlueCloud('temporal_distr_months.png')
    # temporal distribution (seasonally)
    P.temporal_distribution(time_bins='season')
    P.save_BlueCloud('temporal_distr_season.png')
    # save data
    ds.to_netcdf('predicted_dataset.nc', format='NETCDF4')


def main():
    args = get_args()
    var_name_ds = args.var_name_ds
    var_name_mdl = args.var_name_mdl
    features_in_ds = {var_name_mdl: var_name_ds}
    model_path = args.model
    file_name = args.file_name
    print("loading the dataset and model")
    start_time = time.time()
    ds, first_date, coord_dict = load_data(file_name)
    z_dim = coord_dict['depth']
    m = load_model(model_path=model_path)
    load_time = time.time() - start_time
    print("load finished in " + str(load_time) + "sec")
    print("starting predictions and plots")
    start_time = time.time()
    ds = predict(m=m, ds=ds, var_name_mdl=var_name_mdl, var_name_ds=var_name_ds, z_dim=z_dim)
    generate_plots(m=m, ds=ds, var_name_ds=var_name_ds, first_date=first_date)
    train_time = time.time() - start_time
    print("computation finished in " + str(train_time) + "sec")


if __name__ == '__main__':
    main()
