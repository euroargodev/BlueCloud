# Preproseccing functions file
import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def weekly_mean(ds):
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
    #TODO: detect time variable
    #TODO: detect var_name
    
    var_name = 'CHL'
    X = ds.groupby("time.week").mean()
    X = X.rename_dims({'week': 'feature'})
    X = X.rename({'week': 'feature'})
    
    return X
    
def hist_2D(X, bins=None):
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
    #TODO: detect var_name
    var_name = 'CHL'
    
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
    ax.set_xlabel('Weeks')
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

def delate_NaNs():
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

    return BIC, k

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

def unstack_dataset():
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

    # TODO: 

    return BIC, k
