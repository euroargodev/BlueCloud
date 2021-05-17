from OceanPatternsIndicator.Plotter import Plotter
import time
from preprocessing_utils import *
import pandas as pd
import sklearn
from pyxpcm.models import pcm


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
    parse.add_argument('algo', type=str, help='algo choice (Kmean, mini-batch, GMM')
    return parse.parse_args()


def train_model(k, x, var_name_ds, algo):
    if algo == "kmean":
        print("model used: kmean")
        model = sklearn.cluster.KMeans(n_clusters=k, n_init=10, max_iter=1000)
    elif algo == "batch":
        print("model used: mini batch kmean")
        model = sklearn.cluster.MiniBatchKMeans(n_clusters=k, n_init=10, max_iter=1000, batch_size=100)
    else:
        print("model used: GMM")
        model = sklearn.mixture.GaussianMixture(n_components=k, max_iter=1000, tol=1e-6)
    model.fit(x[var_name_ds])
    return model


def predict(x, m, var_name_ds, k, var_predict):
    classif = m.predict(x[var_predict])
    x = x.assign(variables={"labels": ('sample_dim', classif)})
    q = [0.05, 0.5, 0.95]
    x = compute_quantile(x, var_name_ds, k, q)
    x = x.assign_coords(coords={'k': range(k)})
    x = x.unstack('sample_dim')
    return x


def generate_plots(ds, var_name_ds, k, algorithm):
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
    ds : Xarray dataset containing the predictions
    var_name_ds : name of the variable in the dataset

    Returns
    -------
    saves all the plots as png
    """
    try:
        x_label = ds[var_name_ds].attrs['long_name'] + " in " + ds[var_name_ds].attrs['unit_long']
    except KeyError:
        x_label = var_name_ds

    # create a pyXpcm model to use the Plotter class
    var_name_mdl = var_name_ds
    z_dim = 'depth'
    z = ds[z_dim]
    pcm_features = {var_name_mdl: z}
    m = pcm(K=k, features=pcm_features)
    ds = ds.rename({'labels': 'PCM_LABELS'})
    ds = ds.sortby('latitude').sortby('longitude')
    P = Plotter(ds, m, coords_dict={'latitude': 'latitude', 'longitude': 'longitude', 'time': 'time', 'depth': 'depth'})

    # plot profiles by class
    P.vertical_structure(q_variable=var_name_ds + '_Q', sharey=True, xlabel=x_label)
    P.save_BlueCloud('vertical_struc.png')
    # plot profiles by quantile
    P.vertical_structure_comp(q_variable=var_name_ds + '_Q', plot_q='all', xlabel=x_label)
    P.save_BlueCloud('vertical_struc_comp.png')
    # spacial distribution
    P.spatial_distribution(time_slice='most_freq_label')
    P.save_BlueCloud('spatial_distr_freq.png')
    # robustness
    # P.plot_robustness(time_slice=first_date)
    # P.save_BlueCloud('robustness.png')
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
    algorithm = args.algo
    k = args.k
    file_name = args.file_name
    exec_time_log = []
    for i in range(10):
        print("loading the dataset")
        start_time = time.time()
        x = preprocessing_allin(path=file_name, scaling=True, multiple=False, backend='sk', var_name=var_name_ds)
        load_time = time.time() - start_time
        print("load finished in " + str(load_time) + "sec")
        print("starting computation")
        start_time = time.time()
        m = train_model(k=k, x=x, var_name_ds=var_name_ds + "_scaled_reduced", algo=algorithm)
        train_time = time.time() - start_time
        print("training finished in " + str(train_time) + "sec")
        start_time = time.time()
        ds = predict(m=m, x=x, var_name_ds=var_name_ds, var_predict = var_name_ds + "_scaled_reduced", k=k)
        prediction_time = time.time() - start_time
        print("prediction finished in " + str(prediction_time) + "sec")
        start_time = time.time()
        generate_plots(ds=ds, var_name_ds=var_name_ds, k=k, algorithm=algorithm)
        plot_time = time.time() - start_time
        print("plot finished in " + str(plot_time) + "sec")
        tmp_log = {
            'exec_nb': i,
            'ncpu': 8,
            'ram': 16,
            'algorithm': algorithm,
            'platform': "Datarmor",
            'time_load': load_time,
            'time_train': train_time,
            'time_prediction': prediction_time,
            'time_plot': plot_time,
            'total_time': load_time + train_time + prediction_time + plot_time,
            'file_size': (ds.nbytes / 1073741824),
        }
        exec_time_log.append(tmp_log)
    pd.DataFrame(exec_time_log).to_csv("exec_time.csv")
    print("exec time saved")


if __name__ == '__main__':
    main()
