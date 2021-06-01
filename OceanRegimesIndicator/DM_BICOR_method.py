import time
from OceanRegimesIndicator.utils.BIC_calculation_OR import *
from PIL import Image, ImageFont, ImageDraw
from utils.data_loader_utils import *


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
    parse.add_argument('nk', type=int, help='number max of clusters')
    parse.add_argument('var_name_ds', type=str, help='name of variable in dataset')
    parse.add_argument('corr_dist', type=int, help='correlation distance used for BIC')
    return parse.parse_args()


def compute_BIC(ds, var_name_ds, nk, corr_dist):
    """
    The BIC (Bayesian Information Criteria) can be used to optimize the number of classes in the model, trying not to
    over-fit or under-fit the data. To compute this index, the model is fitted to the training dataset for a range of K
     values from 0 to 20. A minimum in the BIC curve will give you the optimal number of classes to be used.
    Parameters
    ----------
    ds : Xarray dataset
    var_name_ds: the name of the variable in the dataset
    nk : number of K to explore (always starts at 1 up to nk)

    Returns
    -------
    bic: all values for the bic graph
    bic_min: min value of the bic
    """
    bic, bic_min = BIC_calculation(X=ds, coords_dict={'latitude': 'lat', 'longitude': 'lon'},
                                   corr_dist=corr_dist,
                                   feature_name='feature_reduced', var_name=var_name_ds + '_reduced',
                                   Nrun=10, NK=nk)
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


def add_2logo(mfname, outfname, ds, logo_height=70, txt_color=(0, 0, 0, 255)):
    """ Add 2 logos and text to a figure

        Parameters
        ----------
        ds : dataset Xarray
        mfname : string
            source figure file
        outfname : string
            output figure file
    """
    font_path = "./utils/logos/Calibri_Regular.ttf"
    lfname2 = "./utils/logos/Blue-cloud_compact_color_W.jpg"
    lfname1 = "./utils/logos/Logo-LOPS_transparent_W.jpg"

    mimage = Image.open(mfname)
    coords_dict = {'latitude': 'lat', 'longitude': 'lon'}
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
    spatial_string = 'Domain: lat:[%.2f,%.2f], lon:[%.2f,%.2f]' % (
        lat_extent[0], lat_extent[1], lon_extent[0], lon_extent[1])

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


def save_bic_plot(bic, nk, ds):
    """
    create and saves the bic plot
    Parameters
    ----------
    bic : bic results
    nk : int, number of k tested (from 1 to nk)
    ds : dataset, Xarray dataset
    """
    out_name = "bic.png"
    plot_BIC(bic, nk)
    plt.savefig(out_name, bbox_inches='tight', pad_inches=0.1)
    # add lower band
    add_lowerband(out_name, out_name)
    # add logo
    add_2logo(out_name, out_name, ds)
    logging.info('Figure saved in %s' % out_name)


def main_BICOR(args):
    """
    Main function of the BIC ocean regimes method
    Parameters
    ----------
    args : Dictionary with:
        file_name: string, dataset path
        nk: int, max range k to use for bic
        var_name_ds: string, name var in dataset
        var_name_mdl: string, name var in model
        corr_dist: int, correlation distance
    """
    var_name_ds = args['var_name']
    corr_dist = args['corr_dist']
    file_name = args['file']
    mask_path = args['mask']
    nk = args['nk']
    arguments_str = f"file_name: {file_name} " \
                    f"var_name_ds: {var_name_ds} " \
                    f"k: {corr_dist}" \
                    f"nk: {nk}" \
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
    bic, bic_min = compute_BIC(ds=ds, var_name_ds=var_name_ds, nk=nk, corr_dist=corr_dist)
    bic_time = time.time() - start_time
    logging.info("bic computation finished in " + str(bic_time) + "sec")
    # plot and save fig
    save_bic_plot(bic=bic, nk=nk, ds=ds_init)


if __name__ == '__main__':
    args = get_args()
    input_dict = {
        'file': args.file_name,
        'mask': args.mask,
        'corr_dist': args.corr_dist,
        'var_name': args.var_name_ds,
        'nk': args.nk
    }
    main_BICOR(input_dict)
