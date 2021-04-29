import time
from data_loader_utils import load_data
from model_train_utils import train_model
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
    parse.add_argument('k', type=int, help="number of clusters K")
    parse.add_argument('file_name', type=str, help='input dataset')
    parse.add_argument('var_name_ds', type=str, help='name of variable in dataset')
    parse.add_argument('var_name_mdl', type=str, help='name of variable in model')

    return parse.parse_args()


def main_fit_predict(args):
    main_start_time = time.time()
    var_name_ds = args.var_name_ds
    var_name_mdl = args.var_name_mdl
    features_in_ds = {var_name_mdl: var_name_ds}
    k = args.k
    file_name = args.file_name
    arguments_str = f"file_name: {file_name} " \
                    f"var_name_ds: {var_name_ds} " \
                    f"var_name_mdl: {var_name_mdl} "
    print(arguments_str)

    # ---------------- Load data --------------- #
    print("loading the dataset")
    start_time = time.time()
    ds, first_date, coord_dict = load_data(file_name=file_name, var_name_ds=var_name_ds)
    z_dim = coord_dict['depth']
    load_time = time.time() - start_time
    print("load finished in " + str(load_time) + "sec")

    # --------- train model -------------- #
    print("starting computation")
    start_time = time.time()
    m = train_model(k=k, ds=ds, var_name_mdl=var_name_mdl, var_name_ds=var_name_ds, z_dim=z_dim)
    train_time = time.time() - start_time
    print("training finished in " + str(train_time) + "sec")

    # ----------- predict ----------- #
    start_time = time.time()
    ds = predict(m=m, ds=ds, var_name_mdl=var_name_mdl, var_name_ds=var_name_ds, z_dim=z_dim)
    ds = robustness(m=m, ds=ds, features_in_ds=features_in_ds, z_dim=z_dim, first_date=first_date)
    ds = quantiles(ds=ds, m=m, var_name_ds=var_name_ds)
    generate_plots(m=m, ds=ds, var_name_ds=var_name_ds, first_date=first_date)
    predict_time = time.time() - start_time
    print("prediction and plots finished in " + str(predict_time) + "sec")
    # save model
    m.to_netcdf('model.nc')
    print("model saved")
    print(f"execution finished in {time.time()-main_start_time}")


if __name__ == '__main__':
    args = get_args()
    main_fit_predict(args)
