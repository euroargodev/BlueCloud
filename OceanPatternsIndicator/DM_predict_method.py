import logging
import time

import pyxpcm

from utils.data_loader_utils import load_data
from utils.prediction_utils import predict, robustness, quantiles, generate_plots
from download.storagehubfacility import storagehubfacility as sthubf, check_json


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


def load_model(model_id):
    """
    load pyXpcm model
    Parameters
    ----------
    model_id : string, id of trained model on storagehub (*.nc)

    Returns
    -------
    pyXpcm trained model
    """
    myshfo = sthubf.StorageHubFacility(operation="Download", ItemId=model_id,
                                       localFile=f'./model{model_id}.nc')
    myshfo.main()

    m = pyxpcm.load_netcdf(f'./model{model_id}.nc')
    logging.info(f"model loaded:\n {m}")
    return m


def main_predict(args):
    """
    Main function of the predict ocean patterns method
    Parameters
    ----------
    args : Dictionary with:
        file: string, dataset path
        model: string, model path
        var_name: string, name var in dataset
        id_field: string, standard name of var
    """
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
    logging.info(f"loadin dataset finished: {ds}")
    z_dim = coord_dict['depth']
    m = load_model(model_id=model_path)
    load_time = time.time() - start_time
    logging.info("load finished in " + str(load_time) + "sec")

    # ------------ predict and plot ----------- #
    logging.info("starting predictions and plots")
    start_time = time.time()
    ds = predict(m=m, ds=ds, var_name_mdl=var_name_mdl, var_name_ds=var_name_ds, z_dim=z_dim)
    ds = robustness(m=m, ds=ds, features_in_ds=features_in_ds, z_dim=z_dim)
    ds = quantiles(ds=ds, m=m, var_name_ds=var_name_ds)
    generate_plots(m=m, ds=ds, var_name_ds=var_name_ds, first_date=first_date)
    predict_time = time.time() - start_time
    logging.info("prediction and plots finished in " + str(predict_time) + "sec")


if __name__ == '__main__':
    args = get_args()
    main_predict(args)
