from sklearn import mixture


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
    model.fit(ds[var_name_ds + "_reduced"])
    return model
