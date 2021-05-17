import logging
import sys
import xarray as xr
import numpy as np
import pyxpcm
from pyxpcm.models import pcm


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
        logging.error("No profiles are deep enough to reach the max depth defined in the dataset, therefore no profiles are left after filtering. Please reduce the max depth of your dataset")
        logging.error(e)
        raise ValueError('')
    return m