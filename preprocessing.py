# Preproseccing functions file
import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

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



def reduce_dims(X):
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

    #TODO: dimensions detection
    sampling_dims = list(X.dims)
    sampling_dims.remove('feature')
    X = X.stack({'sampling': sampling_dims})
    
    return X

def delate_NaNs(ds, X, mask_path='auto'):
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

    # TODO: option use mask or create a mask from dataset
    # TODO: delate all time series totally NaNs
    # TODO: If there are encore the NaNs make interpolation
    # mask should be a boolean dataarray with dimensions lat lon with the same name an same arrays than the dataset
    
    #TODO: detect var_name
    #var_name = 'CHL'
    var_name = 'analysed_sst'
    # TODO: detect coordinates
    sampling_dims = {'lat', 'lon'}
    #check if we have a mask or not
    if 'auto' in mask_path: 
        #create mask
        print('create mask')
        stacked_mask = X[var_name].notnull()
        mask = stacked_mask.unstack('sampling')
    else:
        #use mask
        print('use mask')
        #open mask
        mask = xr.open_dataset(mask_path)
        # TODO: maybe do the format of the mask? or ask for a mask with options? other notebook to crete a mask from a dataset?
        #stack mask
        stacked_mask = mask['mask'].stack({'sampling': sampling_dims})
        
    #apply mask
    X = X[var_name].where(stacked_mask == True, drop=True).to_dataset()
    
    # if lines are totally NaN
    if np.any(np.isnan(X[var_name].values)):
        #delate time series all NaNs
        print('delate time series all NaNs')
        X = X[var_name].where(~X[var_name].isnull(),drop=True).to_dataset()
        
    # interpolation
    if np.any(np.isnan(X[var_name].values)):
        #delate time series all NaNs
        print('interpolation')
        X = X[var_name].interpolate_na(dim = 'feature', method="linear", fill_value="extrapolate").to_dataset(name = var_name)
        
    # check if NaNs in dataset
    if np.any(np.isnan(X[var_name].values)):
        #TODO: warning
        print('Dataset contains NaNs after preprocessing. Please, try the option without mask')


    return X, mask

def scaler(X):
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

    # TODO: option more than one scaler
    #TODO: detect var_name
    var_name = 'CHL'
    
    # Check dimensions order
    X = X.transpose("sampling", "feature")
    
    # Apply Standard Scaler
    from sklearn.preprocessing import StandardScaler
    X_scale = StandardScaler().fit_transform(X[var_name])
    X = X.assign(variables={var_name + "_scaled":(('sampling', 'feature'), X_scale)})

    return X

def apply_PCA(X, plot_var=False):
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

    # TODO: detect var_name
    var_name = 'CHL'
    
    # Check dimensions order
    X = X.transpose("sampling", "feature")
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 0.99, svd_solver = 'full')
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
