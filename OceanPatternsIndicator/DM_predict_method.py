import time

import pyxpcm

from data_loader_utils import load_data
from prediction_utils import predict, robustness, quantiles, generate_plots


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


def load_model(model_path):
    m = pyxpcm.load_netcdf(model_path)
    return m


def main_predict(args):
    var_name_ds = args.var_name_ds
    var_name_mdl = args.var_name_mdl
    features_in_ds = {var_name_mdl: var_name_ds}
    model_path = args.model
    file_name = args.file_name
    print("loading the dataset and model")
    start_time = time.time()
    ds, first_date, coord_dict = load_data(file_name=file_name, var_name_ds=var_name_ds)
    z_dim = coord_dict['depth']
    m = load_model(model_path=model_path)
    load_time = time.time() - start_time
    print("load finished in " + str(load_time) + "sec")
    print("starting predictions and plots")
    start_time = time.time()
    ds = predict(m=m, ds=ds, var_name_mdl=var_name_mdl, var_name_ds=var_name_ds, z_dim=z_dim)
    ds = robustness(m=m, ds=ds, features_in_ds=features_in_ds, z_dim=z_dim, first_date=first_date)
    ds = quantiles(ds=ds, m=m, var_name_ds=var_name_ds)
    generate_plots(m=m, ds=ds, var_name_ds=var_name_ds, first_date=first_date)
    train_time = time.time() - start_time
    print("computation finished in " + str(train_time) + "sec")


if __name__ == '__main__':
    args = get_args()
    main_predict(args)
