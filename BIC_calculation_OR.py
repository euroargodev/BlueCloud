# BIC calculation functions file
import xarray as xr
import numpy as np

import pyxpcm
from pyxpcm.models import pcm

from sklearn import mixture

import matplotlib.pyplot as plt

import concurrent.futures
from tqdm import tqdm

from datetime import datetime

import warnings


def mapping_corr_dist(corr_dist, start_point, grid_extent):
    '''Remapping longitude/latitude grid using a start point. It creates a new grid from the start point 
       where each point is separated the given correlation distance.

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

    # angular distance d/earth's radius (km)
    delta = corr_dist/6371

    # all in radians (conversion at the end)
    grid_extent = grid_extent*np.pi/180
    start_point = start_point*np.pi/180

    ### while loop for lat nord ###
    max_lat = grid_extent[3]
    lat2 = -np.pi/2
    lat1 = start_point[1]
    # bearing = 0 donc cos(0)=1 and sin(0)=0
    new_lats = [lat1]
    while lat2 < max_lat:
        lat2 = np.arcsin(np.sin(lat1)*np.cos(delta) +
                         np.cos(lat1)*np.sin(delta))
        new_lats.append(lat2)
        lat1 = lat2

    ### while loop for lat sud ###
    min_lat = grid_extent[2]
    lat2 = np.pi/2
    lat1 = start_point[1]
    # bearing = pi donc cos(pi)=-1 and sin(pi)=0
    while lat2 > min_lat:
        lat2 = np.arcsin(np.sin(lat1)*np.cos(delta) -
                         np.cos(lat1)*np.sin(delta))
        new_lats.append(lat2)
        lat1 = lat2

    new_lats = np.sort(new_lats)*180/np.pi

    ### while loop for lon east ###
    max_lon = grid_extent[1]
    lon2 = -np.pi
    lon1 = start_point[0]
    lat1 = start_point[1]
    # bearing = pi/2 donc cos(pi/2)=0 and sin(pi/2)=1
    new_lons = [lon1]
    dlon = np.arctan2(np.sin(delta)*np.cos(lat1),
                      np.cos(delta)-np.sin(lat1)*np.sin(lat1))
    while lon2 < max_lon:
        lon2 = lon1 + dlon
        new_lons.append(lon2)
        lon1 = lon2

    ### while loop for lon west ###
    min_lon = grid_extent[0]
    lon2 = np.pi
    lon1 = start_point[0]
    lat1 = start_point[1]
    # bearing = -pi/2 donc cos(-pi/2)=0 and sin(-pi/2)=-1
    dlon = np.arctan2(-np.sin(delta)*np.cos(lat1),
                      np.cos(delta)-np.sin(lat1)*np.sin(lat1))
    while lon2 > min_lon:
        lon2 = lon1 + dlon
        new_lons.append(lon2)
        lon1 = lon2

    new_lons = np.sort(new_lons)*180/np.pi

    return new_lats, new_lons


def mapping_corr_time(corr_time, start_point, time_extent):
    '''Remapping time vector using a start point. It creates a new vector from the start point 
       where elements are separated the time correlation.

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

    # we are supossing that 1 month is 30 days for using np.timedelta64

    ### while loop for bigger dates ###
    max_time = time_extent[1]
    time2 = time_extent[0]
    time1 = start_point[0]
    # bearing = 0 donc cos(0)=1 and sin(0)=0
    new_time = [time1]
    while time2 < max_time:
        time2 = time1 + np.timedelta64(corr_time*30, 'D')
        new_time.append(time2)
        time1 = time2

    ### while loop for smaller dates ###
    min_time = time_extent[0]
    time2 = time_extent[1]
    time1 = start_point[0]
    # bearing = pi donc cos(pi)=-1 and sin(pi)=0
    while time2 > min_time:
        time2 = time1 - np.timedelta64(corr_time*30, 'D')
        new_time.append(time2)
        time1 = time2

    new_time = np.sort(new_time)

    return new_time

def BIC_cal(X, k):
    ''' Function that calculates BIC for a number of classes k

            Parameters
            ----------
                X: dataset after preprocessing
                k: number of classes

            Returns
            ------
                BIC: BIC value
                k: number of classes

            '''

    # create model
    m = mixture.GaussianMixture(n_components=k+1, covariance_type='full')
    # fit model
    m.fit(X)
    # Calculate BIC
    BIC = m.bic(X)

    return BIC, k

def BIC_calculation(X, corr_dist, coords_dict, feature_name, var_name, Nrun=10, NK=20):
    '''Calculation of BIC (Bayesian Information Criteria) for a training dataset.
        The calculation is parallelised using ThreadPoolExecutor.

           Parameters
           ----------
               X: preprocessed dataset
               corr_dist: correlation distance
               coords_dict: dictionary with coordinates names
                    {'depth': 'depth', 'latitude': 'latitude', 'time': 'time', 'longitude': 'longitude'}
               features_name: name of the feature variable
               Nrun: number of runs
               NK: max number of classes

           Returns
           ------
               BIC: matrix with BIC value for each run and number of classes
               BIC_min: minimun BIC value calculated from the mean of each number of classes

               '''

    #start = time.time()
    # TODO: latitude and longitude values
    # TODO: automatic detection of variables names
    # TODO: If only one time step?
    
    #Unstack dataset
    X_unstack = X[var_name].unstack('sampling')
    X_unstack = X_unstack.sortby([coords_dict.get('latitude'),coords_dict.get('longitude')])

    # grid extent
    grid_extent = np.array([X_unstack[coords_dict.get('longitude')].values.min(), X_unstack[coords_dict.get('longitude')].values.max(
    ), X_unstack[coords_dict.get('latitude')].values.min(), X_unstack[coords_dict.get('latitude')].values.max()])

    # this is the list of arguments to iterate over, for instance nb of classes for a PCM
    class_list = np.arange(0, NK)

    

    BIC = np.zeros((NK, Nrun))
    for run in range(Nrun):

        # random fist point
        latp = np.random.choice(X_unstack[coords_dict.get('latitude')].values, 1, replace=False)
        lonp = np.random.choice(X_unstack[coords_dict.get('longitude')].values, 1, replace=False)
        # remapping
        new_lats, new_lons = mapping_corr_dist(
            corr_dist=corr_dist, start_point=np.concatenate((lonp, latp)), grid_extent=grid_extent)        

        ds_run_i = X_unstack.sel({coords_dict.get('latitude'):list(new_lats), coords_dict.get('longitude'):list(new_lons)}, method='nearest')
        X_run_i = ds_run_i.stack({'sampling': ('lat', 'lon')})
        X_run_i = X_run_i.transpose("sampling", feature_name)
        #no NaNs
        X_run_i = X_run_i.where(~X_run_i.isnull(),drop=True).to_dataset()
        
        # BIC computation in parallel
        #results = []
        #ConcurrentExecutor = concurrent.futures.ThreadPoolExecutor(
        #    max_workers=100)
        #with ConcurrentExecutor as executor:
        #    future_to_url = {executor.submit(
        #        BIC_cal, X_run_i[var_name], k): k for k in class_list}
        #   futures = concurrent.futures.as_completed(future_to_url)
        #    futures = tqdm(futures, total=len(class_list))
        #   for future in futures:
        #       traj = None
        #       try:
        #           traj = future.result()
        #        except Exception as e:
        #            # pass
        #            raise
        #        finally:
        #           results.append(traj)
        # Only keep non-empty results
        #results = [r for r in results if r is not None]
        #results.sort(key=lambda x: x[1])
        #BIC[:, run] = np.array([i[0] for i in results])
        
        # serial computation of BIC
        results = []
        for k in class_list:
            results.append(BIC_cal(X_run_i[var_name], k))
            
        results = [r for r in results if r is not None]
        results.sort(key=lambda x: x[1])
        BIC[:, run] = np.array([i[0] for i in results])

    BIC_min = np.argmin(np.mean(BIC, axis=1))+1

    return BIC, BIC_min


def plot_BIC(BIC, NK):
    '''Plot of mean BIC (Bayesian Information Criteria) and standard deviation.

           Parameters
           ----------
               BIC: BIC values obtained using BIC_calculation function
               NK: maximum number of classes

           Returns
           ------
                Figure showing mean 

               '''
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5), dpi=90)
    BICmean = np.mean(BIC, axis=1)
    BICstd = np.std(BIC, axis=1)
    normBICmean = (BICmean-np.mean(BICmean))/np.std(BICmean)
    #normBICstd = np.std(normBICmean)
    #plt.plot(np.arange(kmax)+1,(BIC-np.mean(BIC))/np.std(BIC),label='Raw BIC')
    plt.plot(np.arange(NK)+1, BICmean, label='BIC mean')
    plt.plot(np.arange(NK)+1, BICmean+BICstd,
             color=[0.7]*3, linewidth=0.5, label='BIC std')
    plt.plot(np.arange(NK)+1, BICmean-BICstd, color=[0.7]*3, linewidth=0.5)
    plt.ylabel('BIC')
    plt.xlabel('Number of classes')
    plt.xticks(np.arange(NK)+1)
    plt.legend()
    plt.title('Bayesian information criteria (BIC)')
