import logging
import time

import pyxpcm

from utils.data_loader_utils import load_data
from utils.prediction_utils import predict, robustness, quantiles, generate_plots


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
    var_name_ds = args['var_name']
    var_name_mdl = args['id_field']
    features_in_ds = {var_name_mdl: var_name_ds}
    model_path = args['model']
    file_name = args['file']
    arguments_str = f"\tfile_name: {file_name} \n" \
                    f"\tvar_name_ds: {var_name_ds} \n" \
                    f"\tvar_name_mdl: {var_name_mdl} \n" \
                    f"\tmodel: {model_path} \n"
    logging.info(f"Ocean patterns predict method launched with the following arguments:\n {arguments_str}")

    # ------------ loading data and model ----------- #
    logging.info("loading the dataset and model")
    start_time = time.time()
    ds, first_date, coord_dict = load_data(file_name=file_name, var_name_ds=var_name_ds)
    z_dim = coord_dict['depth']
    m = load_model(model_path=model_path)
    load_time = time.time() - start_time
    logging.info("load finished in " + str(load_time) + "sec")

    # ------------ predict and plot ----------- #
    logging.info("starting predictions and plots")
    start_time = time.time()
    ds = predict(m=m, ds=ds, var_name_mdl=var_name_mdl, var_name_ds=var_name_ds, z_dim=z_dim)
    ds = robustness(m=m, ds=ds, features_in_ds=features_in_ds, z_dim=z_dim, first_date=first_date)
    ds = quantiles(ds=ds, m=m, var_name_ds=var_name_ds)
    generate_plots(m=m, ds=ds, var_name_ds=var_name_ds, first_date=first_date)
    predict_time = time.time() - start_time
    logging.info("prediction and plots finished in " + str(predict_time) + "sec")


if __name__ == '__main__':
    args = get_args()
    main_predict(args)
