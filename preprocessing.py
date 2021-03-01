# Preproseccing functions file
import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import warnings

def weekly_mean(ds, var_name, time_var='auto'):
    '''Weekly mean in dataset

           Parameters
           ----------
               corr_dist: correlation distance
               start_point: latitude and longitude of the start point 
               grid_extent: max and min latitude and longitude of the grid to be remapped 
                    [min lon, max lon, min let, max lat]

           Returns
           ------
               new_lats: new latitude vector with points separeted the correlation distance 
               new_lons: new longitude vector with points separeted the correlation distance

               '''

    #TODO: when more than one year in dataset
    
    #detect time variable from attributes
    if 'auto' in time_var:
        coords_list = list(ds.coords.keys())
        for c in coords_list:
            axis_at = ds[c].attrs.get('axis')
            if axis_at == 'T':
                time_var = c
        if 'auto' in time_var:
            raise ValueError(
                'Time variable could not be detected. Please, provide it using time_var input.')
    
    X = ds.groupby(time_var + ".week").mean()
    X = X.rename_dims({'week': 'feature'})
    X = X.rename({'week': 'feature'})
    
    return X
    
def hist_2D(X, var_name, bins=None, xlabel='Weeks'):
    '''Plot 2D histogram

           Parameters
           ----------
               corr_dist: correlation distance
               start_point: latitude and longitude of the start point 
               grid_extent: max and min latitude and longitude of the grid to be remapped 
                    [min lon, max lon, min let, max lat]

           Returns
           ------
               new_lats: new latitude vector with points separeted the correlation distance 
               new_lons: new longitude vector with points separeted the correlation distance

               '''
    if 'feature' not in list(X.coords.keys()):
            raise ValueError(
                'Dataset should contains feature coordinate. Please, change the name of your feature coordinate to "feature" or use weekly_mean function.')
    
    if np.any(bins) == None: 
        bins = np.linspace(X[var_name].min().values, X[var_name].max().values,num=50)

    histo_2d = [] 
    for iweek in range(np.size(X.feature)):
        hist_values, bin_edges = np.histogram(X[var_name].isel(feature=iweek).values, bins=bins)
        histo_2d.append(hist_values)
    
    fig, ax = plt.subplots(figsize=(12,10))

    plt.pcolormesh(X.feature.values, bins, np.transpose(histo_2d), cmap='Reds', edgecolors='black')
    cbar = plt.colorbar()
    ax.set_ylabel(var_name)
    ax.set_xlabel(xlabel)
    cbar.ax.set_ylabel('Counts')



def reduce_dims(X, sampling_dims='auto'):
    '''Reduce lat lon dimensions to sampling dimension 

           Parameters
           ----------
               corr_time: correlation time
               start_point: date of the start point 
               time_extent: max and min time of the vector to be remapped 
                    [min time, max time]

           Returns
           ------
               new_time: new time vector with points separeted the time correlation

               '''
    
    #detect sampling coordinates from attributes
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

def delate_NaNs(X, var_name, mask_path='auto'):
    ''' Delate NaNs in dataset

            Parameters
            ----------
                X: dataset after preprocessing
                k: number of classes

            Returns
            ------
                BIC: BIC value
                k: number of classes

            '''

    # TODO: input mask should be:
    #       a boolean dataset
    #       variable name should be "mask"
    #       lat and lon dimensition should have the same name than in dataset
    #       lat and lon dimensions should have the same values than in dataset
    # do we ask too many things for the mask? can we format the mask in the function? or should we create another notebook where we can create a mask as we want to be?
    
    if 'sampling' not in list(X.coords.keys()):
            raise ValueError(
                'Dataset should contains sampling coordinate. Please, use function reduce_dims to stack coordinates in you dataset.')
    
    sampling_dims = X.get_index('sampling').names 
    
    #check if we have a mask or not
    if 'auto' in mask_path: 
        #create mask
        stacked_mask = X[var_name].notnull()
        mask = stacked_mask.unstack('sampling').to_dataset()
        mask = mask.rename({var_name: 'mask'})
    else:
        #use mask
        mask = xr.open_dataset(mask_path)
        stacked_mask = mask['mask'].stack({'sampling': sampling_dims})
        
    #apply mask
    X = X[var_name].where(stacked_mask == True, drop=True).to_dataset()
    
    #delate time series all NaNs
    if np.any(np.isnan(X[var_name].values)):
        X = X[var_name].where(~X[var_name].isnull(),drop=True).to_dataset()
        
    # interpolation
    if np.any(np.isnan(X[var_name].values)):
        print('Interpolation is applied')
        if 'feature' not in list(X.coords.keys()):
            raise ValueError(
                'Dataset should contains feature coordinate. Please, change the name of your feature coordinate to "feature" or use weekly_mean function.')
        X = X[var_name].interpolate_na(dim = 'feature', method="linear", fill_value="extrapolate").to_dataset(name = var_name)
        
    # check if NaNs in dataset
    if np.any(np.isnan(X[var_name].values)):
        warnings.warn('Dataset contains NaNs after preprocessing. Please, try the option mask_path="auto"')

    return X, mask

def scaler(X, var_name, scaler_name='StandardScaler'):
    ''' Scale data

            Parameters
            ----------
                X: dataset after preprocessing
                k: number of classes

            Returns
            ------
                BIC: BIC value
                k: number of classes

            '''
    
    if 'sampling' not in list(X.coords.keys()):
            raise ValueError(
                'Dataset should contains sampling coordinate. Please, use function reduce_dims to stack coordinates in you dataset.')
    if 'feature' not in list(X.coords.keys()):
            raise ValueError(
                'Dataset should contains feature coordinate. Please, change the name of your feature coordinate to "feature" or use weekly_mean function.')
    
    # Check dimensions order
    X = X.transpose("sampling", "feature")
    
    # Apply Standard Scaler
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
        
    X = X.assign(variables={var_name + "_scaled":(('sampling', 'feature'), X_scale)})

    return X

def apply_PCA(X, var_name, n_components=0.99, plot_var=False):
    ''' Principal components analysis

            Parameters
            ----------
                X: dataset after preprocessing
                k: number of classes

            Returns
            ------
                BIC: BIC value
                k: number of classes

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
    pca = PCA(n_components = n_components, svd_solver = 'full')
    pca = pca.fit(X[var_name + "_scaled"])
    X_reduced = pca.transform(X[var_name + "_scaled"])
    X = X.assign(variables={var_name + "_reduced":(('sampling', 'feature_reduced'),X_reduced)})
    
    if plot_var:
        fig, ax = plt.subplots()
        pb = plt.bar(range(pca.n_components_), pca.explained_variance_ratio_)
        ax.set_xlabel('n_components')
        ax.set_ylabel('Percentage')
        ax.set_title('Percentage of variance explained by each of the selected components')

    return X

def unstack_dataset(ds, X, mask):
    ''' UNstack dataste

            Parameters
            ----------
                X: dataset after preprocessing
                k: number of classes

            Returns
            ------
                BIC: BIC value
                k: number of classes

            '''

    # TODO: detect var_name
    var_name = 'CHL'
    
    ds_labels = X.unstack('sampling')
    # TODO: maybe sortby is not always necesary
    ds_labels = ds_labels.sortby(['lat','lon'])
    # same lat and lon values in mask and in results
    # the mask we are using or the mask create in delate NaNs function
    # mask = stacked_mask.unstack()
    ds_labels = ds_labels.reindex_like(mask)
    
    #copy atributtes
    ds_labels.attrs = ds.attrs
    #TODO: coordinates recognintion
    ds_labels.lat.attrs = ds.lat.attrs
    ds_labels.lon.attrs = ds.lon.attrs
    #include time coord for save_BlueCloud function
    ds_labels = ds_labels.assign_coords({'time': ds.time.values})
    

    return ds_labels
