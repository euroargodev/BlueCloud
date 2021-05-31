from utils.data_loader_utils import *
from utils.prediction_utils import generate_plots, predict, quantiles, robustness
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
    parse.add_argument('model', type=str, help="input model")
    parse.add_argument('file_name', type=str, help='input dataset')
    parse.add_argument('mask', type=str, help='path to mask')
    parse.add_argument('var_name_ds', type=str, help='name of variable in dataset')
    return parse.parse_args()


def load_model(model_path):
    """
    Load trained model
    Parameters
    ----------
    model_path : path of model to load

    Returns
    -------
    model: trained sklearn GMM model
    k: number of class
    """
    model = joblib.load(model_path)
    k = model.n_components
    return model, k


def main_predictOR(args):
    var_name_ds = args['var_name']
    model_path = args['model']
    file_name = args['file']
    mask_path = args['mask']
    arguments_str = f"file_name: {file_name} " \
                    f"var_name_ds: {var_name_ds} " \
                    f"model: {model_path}" \
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
    model, k = load_model(model_path=model_path)
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


if __name__ == '__main__':
    args = get_args()
    input_dict = {
        'file': args.file_name,
        'mask': args.mask,
        'model': args.model,
        'var_name': args.var_name_ds
    }
    main_predictOR(input_dict)
