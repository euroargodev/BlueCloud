import logging
import time

from OceanPatternsIndicator.Plotter import Plotter
from OceanPatternsIndicator.utils.data_loader_utils import load_data
from OceanPatternsIndicator.utils.model_train_utils import train_model
from OceanPatternsIndicator.utils.prediction_utils import predict, robustness


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


def main_model_fit(args):
    var_name_ds = args['var_name']
    var_name_mdl = args['id_field']
    features_in_ds = {var_name_mdl: var_name_ds}
    k = args['k']
    file_name = args['file']
    arguments_str = f"\tfile_name: {file_name} \n" \
                    f"\tvar_name_ds: {var_name_ds} \n" \
                    f"\tvar_name_mdl: {var_name_mdl} \n" \
                    f"\tk: {k}"
    logging.info(f"Ocean patterns fit method launched with the following arguments:\n {arguments_str}")
    # ----------- loading data ---------- #
    logging.info("loading the dataset")
    start_time = time.time()
    ds, first_date, coord_dict = load_data(file_name=file_name, var_name_ds=var_name_ds)
    z_dim = coord_dict['depth']
    load_time = time.time() - start_time
    logging.info("load finished in " + str(load_time) + "sec")

    # ----------- fitting model ---------- #
    logging.info("starting model fit")
    start_time = time.time()
    m = train_model(k=k, ds=ds, var_name_mdl=var_name_mdl, var_name_ds=var_name_ds, z_dim=z_dim)
    train_time = time.time() - start_time
    logging.info("model fit finished in " + str(train_time) + "sec")

    # ---------- predictions and plot of robustness ------------- #
    ds = predict(m=m, ds=ds, var_name_mdl=var_name_mdl, var_name_ds=var_name_ds, z_dim=z_dim)
    ds = robustness(m=m, ds=ds, features_in_ds=features_in_ds, z_dim=z_dim, first_date=first_date)
    P = Plotter(ds, m)
    P.plot_robustness(time_slice=first_date)
    P.save_BlueCloud('robustness.png')
    logging.info("robustness computation finished, plot saved")
    m.to_netcdf('model.nc')
    logging.info("model saved")


if __name__ == '__main__':
    args = get_args()
    main_model_fit(args)
