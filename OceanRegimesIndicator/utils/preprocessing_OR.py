# Preproseccing functions file
import logging

import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import warnings


def OR_weekly_mean(ds, var_name, time_var='auto'):
    '''Weekly mean in dataset

           Parameters
           ----------
               ds: input dataset
               var_name: variable we want to use
               time_var: name of time varible. Default: 'auto', variable is automatically detected

           Returns
           ------
               X: dataset with a new time dimension 'feature' corresponding to the weekly mean

               '''

    # detect time variable from attributes
    if 'auto' in time_var:
        coords_list = list(ds.coords.keys())
        for c in coords_list:
            axis_at = ds[c].attrs.get('axis')
            if axis_at == 'T':
                time_var = c
        if 'auto' in time_var:
            raise ValueError(
                'Time variable could not be detected. Please, provide it using time_var input.')

    #X = ds.groupby(time_var + ".week").mean()
    X = ds.groupby(ds[time_var].dt.isocalendar().week).mean()
    X = X.rename_dims({'week': 'feature'})
    X = X.rename({'week': 'feature'})

    return X


def OR_hist_2D(X, var_name, bins=None, xlabel='Weeks'):
    '''Plot 2D histogram

           Parameters
           ----------
               X: input dataset. It should contain 'feature' or 'feature_reduced' dimension
               var_name: variable we want to plot 
               bins: bins to be used in the plot. Default: None, 50 bins are used
               xlabel: label in x axis. Default: 'Weeks'

               '''

    if 'feature' not in list(X.coords.keys()) and 'feature_reduced' not in list(X.coords.keys()):
        raise ValueError(
            'Dataset should contains feature coordinate. Please, change the name of your feature coordinate to "feature" or use weekly_mean function.')

    if np.any(bins) == None:
        bins = np.linspace(X[var_name].min().values,
                           X[var_name].max().values, num=50)

    histo_2d = []

    if '_reduced' in var_name:
        feature_name = 'feature_reduced'
        for iweek in range(np.size(X.feature_reduced)):
            hist_values, bin_edges = np.histogram(
                X[var_name].isel(feature_reduced=iweek).values, bins=bins)
            histo_2d.append(hist_values)
    else:
        feature_name = 'feature'
        for iweek in range(np.size(X.feature)):
            hist_values, bin_edges = np.histogram(
                X[var_name].isel(feature=iweek).values, bins=bins)
            histo_2d.append(hist_values)

    fig, ax = plt.subplots(figsize=(12, 10))

    plt.pcolormesh(X[feature_name].values, bins, np.transpose(
        histo_2d), cmap='Reds', edgecolors='black')
    cbar = plt.colorbar()
    ax.set_ylabel(var_name)
    ax.set_xlabel(xlabel)
    cbar.ax.set_ylabel('Counts')


def OR_reduce_dims(X, sampling_dims='auto'):
    '''Reduce latitude and longitude dimensions to sampling dimension 

           Parameters
           ----------
               X: input dataset
               sampling_dims: dimentions to be stacked. Default: 'auto', latitude and longitude dimensions are automatically detected 

           Returns
           ------
               X: stacked dataset with new dimension 'sampling'

               '''

    # detect sampling coordinates from attributes
    if 'auto' in sampling_dims:
        coords_list = list(X.coords.keys())
        sampling_dims = []
        for c in coords_list:
            axis_at = X[c].attrs.get('axis')
            if axis_at == 'Y':
                sampling_dims.append(c)
            if axis_at == 'X':
                sampling_dims.append(c)
        if not sampling_dims:
            raise ValueError(
                'Sampling dimensions could not be detected. Please, provide them using sampling_dims input.')

    X = X.stack({'sampling': sampling_dims})

    return X


def OR_check_mask(X, mask, sampling_dims):
    '''Check if mask can be used 

           Parameters
           ----------
               X: input dataset
               mask: mask 
               sampling_dims: latitude and longitude dimensions names in dataset

           Returns
           ------
               m_ok: if True, mask can be used

               '''

    m_ok = True
    # name of variable should be "mask"
    if 'mask' not in list(mask.keys()):
        m_ok = False
        raise ValueError(
            'Variable in mask should be called "mask".')
    # boolean dataset
    if mask['mask'].values.dtype.name != 'bool':
        m_ok = False
        raise ValueError(
            'Variable in mask should be a boolean array.')
    # lat and lon dims should have the same name in dataset and mask
    if not set(sampling_dims).issubset(set(list(mask.coords.keys()))):
        m_ok = False
        raise ValueError(
            'Coordinates in mask should have the same name than coordinates in dataset.')
    # lat and lon values should be contained in dataset lat and lon values
    if not set(mask[sampling_dims[0]].values).issubset(set(X[sampling_dims[0]].values)) or not set(mask[sampling_dims[1]].values).issubset(set(X[sampling_dims[1]].values)):
        m_ok = False
        raise ValueError(
            'Coordinates values in mask should contained in coordinates values in dataset.')

    return m_ok


def OR_delate_NaNs(X, var_name, mask_path='auto', interp=False):
    ''' Delate NaNs in dataset

            Parameters
            ----------
                X: input dataset. It should be stacked including 'sampling' dimension
                var_name: variable we want to use
                mask_path: path to mask. It should be:
                            - a boolean dataset
                            - variable name should be "mask"
                            - lat and lon dimensition should have the same name than in dataset
                            - lat and lon values should be contained in dataset lat and lon values
                           Default: 'auto', mask is created from input dataset
                interp: if True interpolation is applied. Default: False

            Returns
            ------
                X: dataset without NaNs
                mask: mask used to delate NaNs. It will be used in OR_unstack_dataset function.

            '''

    if 'sampling' not in list(X.coords.keys()):
        raise ValueError(
            'Dataset should contains sampling coordinate. Please, use function reduce_dims to stack coordinates in you dataset.')
    if 'feature' not in list(X.coords.keys()):
        raise ValueError(
            'Dataset should contains feature coordinate. Please, change the name of your feature coordinate to "feature" or use weekly_mean function.')

    sampling_dims = X.get_index('sampling').names

    # check if we have a mask or not
    if 'auto' in mask_path:
        # create mask
        stacked_mask = X[var_name].isel(feature=0).notnull()  # 2D mask
        mask = stacked_mask.unstack('sampling').to_dataset()
        mask = mask.sortby([sampling_dims[0], sampling_dims[1]])
        if 'lon' in sampling_dims[0]:
            mask = mask.transpose(sampling_dims[1], sampling_dims[0], ...)
        mask = mask.rename({var_name: 'mask'})
    else:
        # use mask
        mask = xr.open_dataset(mask_path)
        m_ok = OR_check_mask(X, mask, sampling_dims)
        if m_ok:
            # if mask is smaller than dataset
            mask_extent = [mask[sampling_dims[0]].values.min(), mask[sampling_dims[0]].values.max(
            ), mask[sampling_dims[1]].values.min(), mask[sampling_dims[1]].values.max()]
            dataset_extent = [X[sampling_dims[0]].values.min(), X[sampling_dims[0]].values.max(
            ), X[sampling_dims[1]].values.min(), X[sampling_dims[1]].values.max()]
            # if mask smaller then dataset extent
            if mask_extent != dataset_extent:
                # I need to unstack and stack the dataset: not very performant
                X = X.unstack('sampling')
                X = X.sortby([sampling_dims[0], sampling_dims[1]])
                X = X.sel({sampling_dims[0]: slice(
                    mask_extent[0], mask_extent[1]), sampling_dims[1]: slice(mask_extent[2], mask_extent[3])})
                X = X.stack({'sampling': sampling_dims})

            stacked_mask = mask['mask'].stack({'sampling': sampling_dims})

    # apply mask
    X = X[var_name].where(stacked_mask == True, drop=True).to_dataset()

    # delate time series all NaNs
    if np.any(np.isnan(X[var_name].values)):
        X = X[var_name].where(~X[var_name].isnull(), drop=True).to_dataset()

    # delate time series with any NaN
    if interp:
        # interpolation
        if np.any(np.isnan(X[var_name].values)):
            logging.info('Interpolation is applied')
            X = X[var_name].interpolate_na(
                dim='feature', method="linear", fill_value="extrapolate").to_dataset(name=var_name)
    else:
        # delate time series with any NaN
        if np.any(np.isnan(X[var_name].values)):
            X = X.dropna('sampling', how='any')

    # check if NaNs in dataset
    if np.any(np.isnan(X[var_name].values)):
        warnings.warn(
            'Dataset contains NaNs after preprocessing. Please, try the option mask_path="auto"')

    return X, mask


def OR_scaler(X, var_name, scaler_name='StandardScaler'):
    ''' Scale data

            Parameters
            ----------
                X: input dataset. It should include 'sampling' and 'feature' dimensions
                var_name: variable we want to use
                scaler_name: options are 'StandardScaler', 'Normalizer' and 'MinMaxScaler'. Default: 'StandardScaler' 

            Returns
            ------
                X: dataset including scaled variable

            '''

    if 'sampling' not in list(X.coords.keys()):
        raise ValueError(
            'Dataset should contains sampling coordinate. Please, use function reduce_dims to stack coordinates in you dataset.')
    if 'feature' not in list(X.coords.keys()):
        raise ValueError(
            'Dataset should contains feature coordinate. Please, change the name of your feature coordinate to "feature" or use weekly_mean function.')

    # Check dimensions order
    X = X.transpose("sampling", "feature")

    if 'StandardScaler' in scaler_name:
        from sklearn.preprocessing import StandardScaler
        X_scale = StandardScaler().fit_transform(X[var_name])
    elif 'Normalizer' in scaler_name:
        from sklearn.preprocessing import Normalizer
        X_scale = Normalizer().fit_transform(X[var_name])
    elif 'MinMaxScaler' in scaler_name:
        from sklearn.preprocessing import MinMaxScaler
        X_scale = MinMaxScaler().fit_transform(X[var_name])
    else:
        raise ValueError(
            'scaler_name is not valid. Please, chose between these options: "StandardScaler",  "Normalizer" or "MinMaxScaler".')

    X = X.assign(
        variables={var_name + "_scaled": (('sampling', 'feature'), X_scale)})

    return X


def OR_apply_PCA(X, var_name, n_components=0.99, plot_var=False):
    ''' Principal components analysis

            Parameters
            ----------
                X: input dataset. It should include 'sampling' and 'feature' dimensions
                var_name: variable we want to use
                n_components: percentage of variance to be explained by all components. Default: 0.99
                plot_var: if True, the percentage of variance explained by each of the components is plotted. Default: False.

            Returns
            ------
                X: dataset including reduced variable and new dimension feature_reduced

            '''

    if 'sampling' not in list(X.coords.keys()):
        raise ValueError(
            'Dataset should contains sampling coordinate. Please, use function reduce_dims to stack coordinates in you dataset.')
    if 'feature' not in list(X.coords.keys()):
        raise ValueError(
            'Dataset should contains feature coordinate. Please, change the name of your feature coordinate to "feature" or use weekly_mean function.')

    # Check dimensions order
    X = X.transpose("sampling", "feature")

    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, svd_solver='full')
    pca = pca.fit(X[var_name + "_scaled"])
    X_reduced = pca.transform(X[var_name + "_scaled"])
    X = X.assign(
        variables={var_name + "_reduced": (('sampling', 'feature_reduced'), X_reduced)})

    if plot_var:
        fig, ax = plt.subplots()
        pb = plt.bar(range(pca.n_components_), pca.explained_variance_ratio_)
        ax.set_xlabel('n_components')
        ax.set_ylabel('Percentage')
        ax.set_title(
            'Percentage of variance explained by each of the selected components')

    return X


def OR_unstack_dataset(ds, X, mask, time_var='auto'):
    ''' Unstack dataset and recover attributes

            Parameters
            ----------
                ds: not preprocessed input dataset (or attributtes)
                X: dataset after preprocessing and including model variables
                mask: mask used during preprocesing for delating NaNs
                time_var: name of time variable in ds dataset. Default: 'auto', variable is automatically detected 

            Returns
            ------
                ds_labels: unstack dataset including ds attributes

            '''

    if 'sampling' not in list(X.coords.keys()):
        raise ValueError(
            'Dataset should contains sampling coordinate. Please, use function reduce_dims to stack coordinates in you dataset.')

    sampling_dims = X.get_index('sampling').names
    ds_labels = X.unstack('sampling')
    # same lat and lon values in mask and in results
    ds_labels = ds_labels.reindex_like(mask)
    # sometimes it is necessary to sort lat and lon
    ds_labels = ds_labels.sortby([sampling_dims[0], sampling_dims[1]])

    # copy atributtes from input dataset
    ds_labels.attrs = ds.attrs
    ds_labels[sampling_dims[0]].attrs = ds[sampling_dims[0]].attrs
    ds_labels[sampling_dims[1]].attrs = ds[sampling_dims[1]].attrs

    # detect time variable
    if 'auto' in time_var:
        ds_coords_list = list(ds.coords.keys())
        for c in ds_coords_list:
            axis_at = ds[c].attrs.get('axis')
            if axis_at == 'T':
                time_var = c
        if 'auto' in time_var:
            raise ValueError(
                'Time variable could not be detected. Please, provide it using time_var input.')
    # include time coord for save_BlueCloud function in Plotter_OR class
    ds_labels = ds_labels.assign_coords({'time': ds[time_var].values})
    ds_labels['time'].attrs = ds['time'].attrs

    return ds_labels
