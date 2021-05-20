import logging

import numpy as np
import matplotlib.pyplot as plt
from OceanPatternsIndicator.Plotter import Plotter


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
    return ds


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
    dataset with robustness
    """
    m.predict_proba(ds, features=features_in_ds, dim=z_dim, inplace=True)
    ds.pyxpcm.robustness(m, inplace=True)
    ds.pyxpcm.robustness_digit(m, inplace=True)
    return ds


def quantiles(ds, m, var_name_ds):
    ds = ds.pyxpcm.quantile(m, q=[0.05, 0.5, 0.95], of=var_name_ds, outname=var_name_ds + '_Q', keep_attrs=True,
                            inplace=True)
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
    P.save_BlueCloud('vertical_struct.png')
    # plot profiles by quantile
    P.vertical_structure_comp(q_variable=var_name_ds + '_Q', plot_q='all', xlabel=x_label, ylim=[-1000, 0])
    P.save_BlueCloud('vertical_struct_comp.png')
    # spacial distribution
    P.spatial_distribution(time_slice='most_freq_label')
    P.save_BlueCloud('spatial_dist_freq.png')
    # robustness
    P.plot_robustness(time_slice=first_date)
    P.save_BlueCloud('robustness.png')
    # pie chart of the classes distribution
    P.pie_classes()
    P.save_BlueCloud('pie_chart.png')
    # temporal distribution (monthly)
    try:
        P.temporal_distribution(time_bins='month')
        P.save_BlueCloud('temporal_dist_months.png')
    except (ValueError, AssertionError) as e:
        save_empty_plot('temporal_dist_months')
        logging.warning('plot monthly temporal distribution is not available, the following error occurred:')
        logging.exception(e)
    # temporal distribution (seasonally)
    try:
        P.temporal_distribution(time_bins='season')
        P.save_BlueCloud('temporal_dist_season.png')
    except (ValueError, AssertionError) as e:
        save_empty_plot('temporal_dist_season')
        logging.warning('plot seasonal temporal distribution is not available, the following error occurred:')
        logging.exception(e)
    # save data
    ds.to_netcdf('predicted_dataset.nc', format='NETCDF4')