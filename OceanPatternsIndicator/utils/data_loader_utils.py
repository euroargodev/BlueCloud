import numpy as np
import xarray as xr
import logging


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
    ds = xr.open_mfdataset(file_name).load()
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
        logging.warning(f"{e} dimension was missing,it has been initialized to 0 for surface data")
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