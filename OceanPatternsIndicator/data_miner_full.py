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
import matplotlib.pyplot as plt
from tools import json_builder
from dateutil.tz import tzutc
from datetime import datetime


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
    ds = ds[[var_name_ds]]
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
    Train a pyXpcm model

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
    """
    # create model
    z = ds[z_dim]
    pcm_features = {var_name_mdl: z}
    m = pcm(K=k, features=pcm_features, maxvar=15)
    # fit model
    features_in_ds = {var_name_mdl: var_name_ds}
    try:
        m.fit(ds, features_in_ds, dim=z_dim)
    except ValueError as e:
        print("No profiles are deep enough to reach the max depth defined in the dataset, therefore no profiles are left after filtering. Please reduce the max depth of your dataset")
        print(e, file=sys.stderr)
        raise ValueError('')
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
    try:
        x_label = ds[var_name_ds].attrs['long_name'] + " in " + ds[var_name_ds].attrs['unit_long']
    except KeyError:
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
    try:
        P.temporal_distribution(time_bins='month')
        P.save_BlueCloud('temporal_distr_months.png')
    except (ValueError, AssertionError) as e:
        save_empty_plot('temporal_distr_months')
        print('plot monthly temporal distribution is not available, the following error occurred:')
        print(e, file=sys.stderr)
    # temporal distribution (seasonally)
    try:
        P.temporal_distribution(time_bins='season')
        P.save_BlueCloud('temporal_distr_season.png')
    except (ValueError, AssertionError) as e:
        save_empty_plot('temporal_distr_season')
        print('plot seasonal temporal distribution is not available, the following error occurred:')
        print(e, file=sys.stderr)
    # save data
    ds.to_netcdf('predicted_dataset.nc', format='NETCDF4')


def get_iso_timestamp():
    isots = datetime.now(tz=tzutc()).replace(microsecond=0).isoformat()
    return isots


def error_exit(err_log, exec_log):
    """
    This function is called if there's an error occurs, it write in log_err the code error with
    a relative message, then copy some mock files in order to avoid bluecloud to terminate with error
    """
    # shutil.copy("./mock/output.nc", "output.nc")
    # shutil.copy("./mock/output.png", "output.png")
    end_time = get_iso_timestamp()
    json_builder.write_json(error=err_log.__dict__,
                            exec_info=exec_log.__dict__['messages'],
                            end_time=end_time)
    exit(1)


def main():
    main_start_time = time.time()
    args = get_args()
    var_name_ds = args.var_name_ds
    var_name_mdl = args.var_name_mdl
    features_in_ds = {var_name_mdl: var_name_ds}
    k = args.k
    file_name = args.file_name
    arguments_str = f"file_name: {file_name} " \
                    f"var_name_ds: {var_name_ds} " \
                    f"var_name_mdl: {var_name_mdl} "
    print(arguments_str)
    exec_log = json_builder.get_exec_log()
    exec_log.add_message(f"BIC methode was launched with the following arguments: {arguments_str}")

    # ---------------- Load data --------------- #
    exec_log.add_message("Start loading dataset")
    print("loading the dataset")
    start_time = time.time()
    ds, first_date, coord_dict = load_data(file_name=file_name, var_name_ds=var_name_ds)
    z_dim = coord_dict['depth']
    load_time = time.time() - start_time
    exec_log.add_message("Loading dataset complete", load_time)
    print("load finished in " + str(load_time) + "sec")

    # --------- train model -------------- #
    print("starting computation")
    exec_log.add_message("Starting model train")
    start_time = time.time()
    try:
        m = train_model(k=k, ds=ds, var_name_mdl=var_name_mdl, var_name_ds=var_name_ds, z_dim=z_dim)
    except ValueError as e:
        err_log = json_builder.LogError(-1, "No profiles are deep enough to reach the max depth defined in the dataset, "
                                            "therefore no profiles are left after filtering. Please reduce the max depth"
                                            " of your dataset" + str(e))
        error_exit(err_log, exec_log)
    train_time = time.time() - start_time
    exec_log.add_message("training complete", train_time)
    print("training finished in " + str(train_time) + "sec")

    # ----------- predict ----------- #
    exec_log.add_message("Starting prediction and plotting")
    start_time = time.time()
    ds = predict(m=m, ds=ds, var_name_mdl=var_name_mdl, var_name_ds=var_name_ds, z_dim=z_dim)
    generate_plots(m=m, ds=ds, var_name_ds=var_name_ds, first_date=first_date)
    predict_time = time.time() - start_time
    exec_log.add_message("predict and plot complete", predict_time)
    print("prediction and plots finished in " + str(predict_time) + "sec")
    # save model
    m.to_netcdf('model.nc')
    print("model saved")
    exec_log.add_message("model saved")
    # Save info in json file
    exec_log.add_message("Total time: " + " %s seconds " % (time.time() - main_start_time))
    err_log = json_builder.LogError(0, "Execution Done")
    end_time = get_iso_timestamp()
    json_builder.write_json(error=err_log.__dict__,
                            exec_info=exec_log.__dict__['messages'],
                            end_time=end_time)



if __name__ == '__main__':
    main()
