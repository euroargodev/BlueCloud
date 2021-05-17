import logging
import time

from PIL import Image, ImageFont, ImageDraw

from OceanPatternsIndicator.utils.BIC_calculation import *
from OceanPatternsIndicator.utils.data_loader_utils import *


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


def save_bic_plot(bic, nk, ds, coords_dict, bic_min):
    out_name = "BIC.png"
    plot_BIC(bic, nk, bic_min=bic_min)
    plt.savefig(out_name, bbox_inches='tight', pad_inches=0.1)
    # add lower band
    add_lowerband(out_name, out_name)
    # add logo
    add_2logo(out_name, out_name, ds, coords_dict)
    print('Figure saved in %s' % out_name)


def main_bic_computation(args):
    file_name = './datasets/*.nc'
    nk = args['nk']
    var_name_ds = args['var_name']
    var_name_mdl = args['var_name']
    corr_dist = args['corr_dist']
    features_in_ds = {var_name_mdl: var_name_ds}
    arguments_str = f"file_name: {file_name} " \
                    f"nk: {nk}" \
                    f"var_name_ds: {var_name_ds} " \
                    f"var_name_mdl: {var_name_mdl} " \
                    f"corr_dist: {corr_dist} "
    logging.info(f"Ocean patterns BIC method launched with the following arguments:\n {arguments_str}")

    # ---------------- Load data --------------- #
    logging.info("loading the dataset")
    start_time = time.time()
    ds, first_date, coord_dict = load_data(file_name=file_name, var_name_ds=var_name_ds)
    z_dim = coord_dict['depth']
    load_time = time.time() - start_time
    logging.info("load finished in " + str(load_time) + "sec")

    # -------------- BIC computation ----------#
    logging.info("starting computation")
    start_time = time.time()
    bic, bic_min = bic_calculation(ds=ds, features_in_ds=features_in_ds, z_dim=z_dim, var_name_mdl=var_name_mdl, nk=nk,
                                   corr_dist=corr_dist, coord_dict=coord_dict, first_date=first_date)
    bic_time = time.time() - start_time
    logging.info("computation finished in " + str(bic_time) + "sec")

    # ---------- Plot BIC -----------------#
    logging.info("Starting BIC plot")
    save_bic_plot(bic=bic, nk=nk, ds=ds, coords_dict=coord_dict, bic_min=bic_min)
    logging.info("Plotting complete, file saved")


if __name__ == '__main__':
    args = get_args()
    main_bic_computation(args)
