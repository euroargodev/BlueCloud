import sys
import xarray as xr
import numpy as np
import pyxpcm
from pyxpcm.models import pcm
import Plotter
from Plotter import Plotter
from BIC_calculation import *
import dask
import dask.array as da
import time
from PIL import Image, ImageFont, ImageDraw
from tools import json_builder
from dateutil.tz import tzutc
from datetime import datetime


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
    parse.add_argument('nk', type=int, help='number max of clusters')
    parse.add_argument('var_name_ds', type=str, help='name of variable in dataset')
    parse.add_argument('var_name_mdl', type=str, help='name of variable in model')
    parse.add_argument('corr_dist', type=int, help='correlation distance used for BIC')
    return parse.parse_args()


def error_exit(err_log, exec_log):
    """
    This function is called if there's an error occurs, it write in log_err the code error with
    a relative message, then copy some mock files in order to avoid bluecloud to terminate with error
    """
    end_time = get_iso_timestamp()
    json_builder.write_json(error=err_log.__dict__,
                            exec_info=exec_log.__dict__['messages'],
                            end_time=end_time)
    exit(0)


def get_iso_timestamp():
    isots = datetime.now(tz=tzutc()).replace(microsecond=0).isoformat()
    return isots


def load_data(file_name, var_name_ds):
    """
    Load dataset into a Xarray dataset

    Parameters
    ----------
    var_name_ds : name of variable in dataset
    file_name : Path to the NetCDF dataset

    Returns
    -------
    ds: Xarray dataset
    first_date: string, first time slice of the dataset
    coord_dict: coordinate dictionary for pyXpcm
    """
    ds = xr.open_dataset(file_name)
    # select var
    ds = ds[[var_name_ds]]
    first_date = str(ds.time.min().values)[0:7]
    # exception to handle missing depth dim: setting depth to 0 because the dataset most likely represents surface data
    try:
        coord_dict = get_coords_dict(ds)
        ds['depth'] = -np.abs(ds[coord_dict['depth']].values)
        ds.depth.attrs['axis'] = 'Z'
    except KeyError as e:
        ds = ds.expand_dims('depth').assign_coords(depth=("depth", [0]))
        ds.depth.attrs['axis'] = 'Z'
        coord_dict = get_coords_dict(ds)
        print(f"{e} dimension was missing,it has been initialized to 0 for surface data", file=sys.stderr)
        err_log = json_builder.LogError(-1, str(e))
    return ds, first_date, coord_dict


def get_coords_dict(ds):
    """
    create a dict of coordinates to mapping each dimension of the dataset
    Parameters
    ----------
    ds : Xarray dataset

    Returns
    -------
    coords_dict: dict mapping each dimension of the dataset
    """
    # creates dictionary with coordinates
    coords_list = list(ds.coords.keys())
    coords_dict = {}
    for c in coords_list:
        axis_at = ds[c].attrs.get('axis')
        if axis_at == 'Y':
            coords_dict.update({'latitude': c})
        if axis_at == 'X':
            coords_dict.update({'longitude': c})
        if axis_at == 'T':
            coords_dict.update({'time': c})
        if axis_at == 'Z':
            coords_dict.update({'depth': c})
    return coords_dict


def bic_calculation(ds, features_in_ds, z_dim, var_name_mdl, nk, corr_dist, coord_dict, first_date):
    """
    The BIC (Bayesian Information Criteria) can be used to optimize the number of classes in the model, trying not to
    over-fit or under-fit the data. To compute this index, the model is fitted to the training dataset for a range of K
     values from 0 to 20. A minimum in the BIC curve will give you the optimal number of classes to be used.
    Parameters
    ----------
    ds : Xarray dataset
    features_in_ds : dict {var_name_mdl: var_name_ds} with var_name_mdl the name of the variable in the model and
    var_name_ds the name of the variable in the dataset
    z_dim : z axis dimension (depth)
    var_name_mdl : name of the variable in the model
    nk : number of K to explore (always starts at 1 up to nk)

    Returns
    -------
    bic: all values for the bic graph
    bic_min: min value of the bic
    """
    z = ds[z_dim]
    pcm_features = {var_name_mdl: z}
    # TODO choose one time frame if short or choose one winter/summer pair
    time_steps = [first_date]
    # time_steps = ['2018-01', '2018-07']  # time steps to be used into account
    nrun = 10  # number of runs for each k
    bic, bic_min = BIC_calculation(ds=ds, coords_dict=coord_dict,
                                   corr_dist=corr_dist, time_steps=time_steps,
                                   pcm_features=pcm_features, features_in_ds=features_in_ds, z_dim=z_dim,
                                   Nrun=nrun, NK=nk)
    return bic, bic_min


def add_lowerband(mfname, outfname, band_height=70, color=(255, 255, 255, 255)):
    """ Add lowerband to a figure

        Parameters
        ----------
        mfname : string
            source figure file
        outfname : string
            output figure file
    """
    image = Image.open(mfname, 'r')
    image_size = image.size
    width = image_size[0]
    height = image_size[1]
    background = Image.new('RGBA', (width, height + band_height), color)
    background.paste(image, (0, 0))
    background.save(outfname)


def add_2logo(mfname, outfname, ds, coords_dict, logo_height=70, txt_color=(0, 0, 0, 255)):
    """ Add 2 logos and text to a figure

        Parameters
        ----------
        coords_dict : coordinates dictionary (see get_coords_dict)
        ds : dataset Xarray
        mfname : string
            source figure file
        outfname : string
            output figure file
    """
    font_path = "./logos/Calibri_Regular.ttf"
    lfname2 = "./logos/Blue-cloud_compact_color_W.jpg"
    lfname1 = "./logos/Logo-LOPS_transparent_W.jpg"

    mimage = Image.open(mfname)
    # Open logo images:
    limage1 = Image.open(lfname1)
    limage2 = Image.open(lfname2)

    # Resize logos to match the requested logo_height:
    aspect_ratio = limage1.size[1] / limage1.size[0]  # height/width
    simage1 = limage1.resize((int(logo_height / aspect_ratio), logo_height))

    aspect_ratio = limage2.size[1] / limage2.size[0]  # height/width
    simage2 = limage2.resize((int(logo_height / aspect_ratio), logo_height))

    # Paste logos along the lower white band of the main figure:
    box = (0, mimage.size[1] - logo_height)
    mimage.paste(simage1, box)

    box = (simage1.size[0], mimage.size[1] - logo_height)
    mimage.paste(simage2, box)

    # Add dataset and model information
    # time extent
    if 'time' not in ds.coords:
        time_string = 'Period: Unknown'
    elif len(ds['time'].sizes) == 0:
        # TODO: when using isel hours information is lost
        time_extent = ds['time'].dt.strftime("%Y/%m/%d %H:%M")
        time_string = 'Period: %s' % time_extent.values
    else:
        time_extent = [min(ds['time'].dt.strftime(
            "%Y/%m/%d")), max(ds['time'].dt.strftime("%Y/%m/%d"))]
        time_string = 'Period: from %s to %s' % (
            time_extent[0].values, time_extent[1].values)

    # spatial extent
    lat_extent = [min(ds[coords_dict.get('latitude')].values), max(
        ds[coords_dict.get('latitude')].values)]
    lon_extent = [min(ds[coords_dict.get('longitude')].values), max(
        ds[coords_dict.get('longitude')].values)]
    spatial_string = 'Domain: lat:%s, lon:%s' % (
        str(lat_extent), str(lon_extent))

    txtA = "User selection:\n   %s\n   %s\n   %s\nSource: %s" % (ds.attrs.get(
        'title'), time_string, spatial_string, 'CMEMS')

    fontA = ImageFont.truetype(font_path, 10)
    txtsA = fontA.getsize_multiline(txtA)
    xoffset = 5 + simage1.size[0] + simage2.size[0]
    posA = (xoffset, mimage.size[1] - txtsA[1] - 5)

    # Print
    drawA = ImageDraw.Draw(mimage)
    drawA.text(posA, txtA, txt_color, font=fontA)

    # Final save
    mimage.save(outfname)


def save_bic_plot(bic, nk, ds, coords_dict):
    out_name = "BIC.png"
    plot_BIC(bic, nk)
    plt.savefig(out_name, bbox_inches='tight', pad_inches=0.1)
    # add lower band
    add_lowerband(out_name, out_name)
    # add logo
    add_2logo(out_name, out_name, ds, coords_dict)
    print('Figure saved in %s' % out_name)


def main():
    main_start_time = time.time()
    args = get_args()
    file_name = args.file_name
    nk = args.nk
    var_name_ds = args.var_name_ds
    var_name_mdl = args.var_name_mdl
    corr_dist = args.corr_dist
    features_in_ds = {var_name_mdl: var_name_ds}
    arguments_str = f"file_name: {file_name} " \
                    f"nk: {nk}" \
                    f"var_name_ds: {var_name_ds} " \
                    f"var_name_mdl: {var_name_mdl} " \
                    f"corr_dist: {corr_dist} "
    print(arguments_str)
    exec_log = json_builder.get_exec_log()
    exec_log.add_message(f"BIC methode was launched with the following arguments: {arguments_str}")
    # ---------------- Load data --------------- #
    exec_log.add_message("Start loading dataset")
    print("loading the dataset")
    start_time = time.time()
    ds, first_date, coord_dict = load_data(file_name=file_name, var_name_ds=var_name_ds)
    z_dim = coord_dict['depth']
    load_time = time.time() - start_time
    exec_log.add_message("Loading dataset complete", load_time)
    print("load finished in " + str(load_time) + "sec")
    # -------------- BIC computation ----------#
    exec_log.add_message("Starting BIC computation")
    print("starting computation")
    start_time = time.time()
    bic, bic_min = bic_calculation(ds=ds, features_in_ds=features_in_ds, z_dim=z_dim, var_name_mdl=var_name_mdl, nk=nk,
                                   corr_dist=corr_dist, coord_dict=coord_dict, first_date=first_date)
    bic_time = time.time() - start_time
    exec_log.add_message("BIC computation complete", bic_time)
    exec_log.add_message(f"bic min = {bic_min}")
    print("computation finished in " + str(bic_time) + "sec")
    # ---------- Plot BIC -----------------#
    exec_log.add_message("Starting BIC plot")
    plot_BIC(BIC=bic, NK=nk, bic_min=bic_min)
    print("plot finished, saving png")
    save_bic_plot(bic=bic, nk=nk, ds=ds, coords_dict=coord_dict)
    exec_log.add_message("Plotting complete, file saved")
    # Save info in json file
    exec_log.add_message("Total time: " + " %s seconds " % (time.time() - main_start_time))
    err_log = json_builder.LogError(0, "Execution Done")
    end_time = get_iso_timestamp()
    json_builder.write_json(error=err_log.__dict__,
                            exec_info=exec_log.__dict__['messages'],
                            end_time=end_time)


if __name__ == '__main__':
    main()
