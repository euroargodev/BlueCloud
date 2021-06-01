from utils.data_loader_utils import *
from utils.model_train_utils import train_model
from utils.prediction_utils import quantiles, robustness, predict, generate_plots
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


def main_fitpred_OR(args):
    """
    Main function of the fit predict ocean regimes method
    Parameters
    ----------
    args : Dictionary with:
        file: string, dataset path
        k: int, number of class
        var_name: string, name var in dataset
        id_field: string, standard name of var
        mask: string, path to mask or 'auto'
    """
    var_name_ds = args['var_name']
    k = args['k']
    file_name = args['file']
    mask_path = args['mask']
    arguments_str = f"file_name: {file_name} " \
                    f"var_name_ds: {var_name_ds} " \
                    f"k: {k}" \
                    f"mask: {mask_path}"
    logging.info(f"Ocean patterns fit predict method launched with the following arguments:\n {arguments_str}")

    logging.info("loading the dataset")
    start_time = time.time()
    ds_init = load_data(file_name=file_name, var_name_ds=var_name_ds)
    load_time = time.time() - start_time
    logging.info("load finished in " + str(load_time) + "sec")

    logging.info("preprocess the dataset")
    start_time = time.time()
    ds, mask = preprocessing_ds(ds=ds_init, var_name_ds=var_name_ds, mask_path=mask_path)
    load_time = time.time() - start_time
    logging.info("preprocessing finished in " + str(load_time) + "sec")

    logging.info("starting computation")
    start_time = time.time()
    model = train_model(k=k, ds=ds, var_name_ds=var_name_ds)
    train_time = time.time() - start_time
    logging.info("training finished in " + str(train_time) + "sec")

    logging.info("starting predictions")
    start_time = time.time()
    ds = predict(model=model, ds=ds, var_name_ds=var_name_ds)
    ds = robustness(model=model, ds=ds, var_name_ds=var_name_ds)
    ds = quantiles(ds=ds, var_name_ds=var_name_ds, k=k, mask=mask, ds_init=ds_init)
    predict_time = time.time() - start_time
    logging.info("prediction finished in " + str(predict_time) + "sec")

    start_time = time.time()
    generate_plots(model=model, ds=ds, var_name_ds=var_name_ds)
    plot_time = time.time() - start_time
    logging.info("plots finished in " + str(plot_time) + "sec")

    # save model
    joblib.dump(model, 'modelOR.sav')
    logging.info("model saved in  modelOR.sav")


if __name__ == '__main__':
    args = get_args()
    input_dict = {
        'file': args.file_name,
        'mask': args.mask,
        'k': args.k,
        'var_name': args.var_name_ds
    }
    main_fitpred_OR(input_dict)
