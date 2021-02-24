# Preproseccing functions file
import xarray as xr
import numpy as np
import pandas as pd


def weekly_mean():
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
    #TODO: option 2D plot
    #TODO: change 'week' to feature
    
    return new_lats, new_lons


def reduce_dims():
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
    
    return new_time

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

def scaler():
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

    return BIC, k

def PCA():
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

    # TODO: at each time be sur that the dataset is in the right order
    # TODO: option plot

    return BIC, k

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
