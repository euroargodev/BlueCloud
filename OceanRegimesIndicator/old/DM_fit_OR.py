from sklearn import mixture
from OceanRegimesIndicator.utils.preprocessing_OR import *
from OceanRegimesIndicator.utils import Plotter_OR
from OceanRegimesIndicator.utils.Plotter_OR import Plotter_OR
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
    parse.add_argument('file_name', type=str, help='input dataset')
    parse.add_argument('mask', type=str, help='path to mask')
    parse.add_argument('k', type=int, help="number of clusters K")
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
    if not np.issubdtype(ds.indexes['time'].dtype, np.datetime64):
        print("casting time to datetimeindex")
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


def train_model(k, ds, var_name_ds):
    """
    Train a pyXpcm model

    Parameters
    ----------
    k : number of clusters
    ds : Xarray dataset
    var_name_ds : name of variable in dataset

    Returns
    -------
    model: Trained model
    """
    model = mixture.GaussianMixture(n_components=k, covariance_type='full', max_iter=500, tol=1e-6, n_init=1)
    x_labels = model.fit_predict(ds[var_name_ds + "_reduced"])
    ds = ds.assign(variables={"GMM_labels": ('sampling', x_labels)})
    return model, ds


def generate_dev_plots(ds, model, var_name_ds, ds_init, mask):
    """
    Robustness: Robustness is a scaled probability of a time series to belong to a class. When looking at the spatial
    distribution of the robustness metric, and if classes have a spatial structure, you may encounter regions with high
    probabilities: these regions are the "core" of the class.

    Parameters
    ----------
    mask : mask used to delete NaNs, used for unstack
    ds_init : Initial dataset to keep all the attributes
    model : trained model
    ds : Xarray dataset containing the predictions
    var_name_ds : name of the variable in the dataset
    Returns
    -------
    saves all the plots as png
    """
    x_proba = model.predict_proba(ds[var_name_ds + "_reduced"])
    ds = ds.assign(variables={"GMM_post": (('sampling', 'k'), x_proba)})
    # calculate robustness
    maxpost = ds["GMM_post"].max(dim="k")
    nk = len(ds["GMM_labels"])
    robust = (maxpost - 1. / nk) * nk / (nk - 1.)
    plist = [0, 0.33, 0.66, 0.9, .99, 1]
    rowl0 = ('Unlikely', 'As likely as not', 'Likely', 'Very Likely', 'Virtually certain')
    robust_id = np.digitize(robust, plist) - 1
    ds = ds.assign(variables={"GMM_robustness": ('sampling', robust), "GMM_robustness_cat": ('sampling', robust_id)})
    ds["GMM_robustness_cat"].attrs['legend'] = rowl0
    ds = OR_unstack_dataset(ds_init, ds, mask)

    P = Plotter_OR(ds, model)
    P.scatter_PDF(var_name=var_name_ds + '_reduced')
    P.save_BlueCloud('scatter_PDF.png')
    # robustness
    P.plot_robustness()
    P.save_BlueCloud('robustness.png')


def main():
    args = get_args()
    var_name_ds = args.var_name_ds
    k = args.k
    file_name = args.file_name
    mask_path = args.mask

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
    model, ds = train_model(k=k, ds=ds, var_name_ds=var_name_ds)
    train_time = time.time() - start_time
    print("training finished in " + str(train_time) + "sec")

    start_time = time.time()
    generate_dev_plots(model=model, ds=ds, var_name_ds=var_name_ds, ds_init=ds_init, mask=mask)
    plot_time = time.time() - start_time
    print("plots finished in " + str(plot_time) + "sec")

    # save model
    joblib.dump(model, 'modelOR.sav')
    print("model saved")


if __name__ == '__main__':
    main()
