#!/usr/bin/env python

import time
import psutil
import multiprocessing as mp
import preprocessing_utils as pre_u
import pandas as pd
import seaborn as sns
import glob
import dask
import numpy as np
import dask.array as da
from dask import delayed
from dask.distributed import Client


def get_args():
    """
    Extract arguments from command line

    Returns
    -------
    parse.parse_args(): dict of the arguments

    """
    import argparse

    parse = argparse.ArgumentParser(description="Ocean patterns method")
    parse.add_argument('n_core', type=int, help='number core')
    parse.add_argument('mem_size', type=int, help='RAM available')
    return parse.parse_args()


def load_ds(file_path, var_name, ncpu, mem):
    print("loading the dataset")
    load_log = []
    test_list = [1, 2, 3, 5, 10, 15, 20, 30, 31]

    for nb_files in test_list:
        # read DS and select var
        start_time = time.time()
        ds_full = pre_u.read_dataset(file_path[:nb_files], multiple=True, backend='dask')
        ds = pre_u.select_var(ds_full, var_name, multiple=True, backend='dask')
        load_time = time.time() - start_time
        print(f"load finished in {load_time} sec for {nb_files} nb files")

        # filter profiles and formatting
        start_time = time.time()
        x = pre_u.filter_profiles(ds)
        filter_time = time.time() - start_time
        print("filter finished in " + str(filter_time) + "sec")

        start_time = time.time()
        x = pre_u.reformat_depth(x)
        x = x.transpose()
        reformat_time = time.time() - start_time

        tmp_log = {
            'ncpu': ncpu,
            'ram': mem,
            'nb_file': nb_files,
            'time_load': load_time,
            'time_filter': filter_time,
            'reformat_time': reformat_time,
            'total_time': load_time + filter_time + reformat_time,
            'full_file_size': (ds_full.nbytes / 1073741824),
            'size_var_sel': (ds.nbytes / 1073741824),
            'final_size': (x.nbytes / 1073741824)
        }
        load_log.append(tmp_log)
        del ds_full, ds, x, tmp_log, start_time, load_time, filter_time
    df_load = pd.DataFrame(load_log)
    return df_load


def main():
    args = get_args()
    ncpu = args.n_core
    mem = args.mem_size
    dask.config.set(temporary_directory='/home1/scratch/lbachelo/')
    client = Client()
    #     client = Client(n_workers=ncpu)
    print("number of core max: " + str(ncpu))
    print(f"{mem} go of ram available")
    file_list = glob.glob("/home/ref-ocean-reanalysis/global-reanalysis-phy-001-030-daily/2018/01/*.nc")
    var_name = 'thetao'
    df_load = load_ds(file_list, var_name, ncpu, mem)
    print(df_load)
    df_load.to_csv(f'./log_out/v2load_log_{ncpu}C_{mem}R.csv')
    client.shutdown()


if __name__ == '__main__':
    main()
