# BIC calculation functions file
import xarray as xr
import numpy as np

import pyxpcm
from pyxpcm.models import pcm

import matplotlib.pyplot as plt

import concurrent.futures
from tqdm import tqdm

def mapping_corr_dist(corr_dist, start_point, grid_extent):
    # function remapping grid using start point and grid extent

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
        lat2 = np.arcsin(np.sin(lat1)*np.cos(delta) + np.cos(lat1)*np.sin(delta))
        new_lats.append(lat2)
        lat1 = lat2

        
    ### while loop for lat sud ###
    min_lat = grid_extent[2]
    lat2 = np.pi/2 
    lat1 = start_point[1]
    # bearing = pi donc cos(pi)=-1 and sin(pi)=0
    while lat2 > min_lat:
        lat2 = np.arcsin(np.sin(lat1)*np.cos(delta) - np.cos(lat1)*np.sin(delta))
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
    dlon = np.arctan2(np.sin(delta)*np.cos(lat1), np.cos(delta)-np.sin(lat1)*np.sin(lat1))
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
    dlon = np.arctan2(-np.sin(delta)*np.cos(lat1), np.cos(delta)-np.sin(lat1)*np.sin(lat1))
    while lon2 > min_lon:
        lon2 = lon1 + dlon
        new_lons.append(lon2)
        lon1 = lon2
        
    new_lons = np.sort(new_lons)*180/np.pi
    
    return new_lats, new_lons



def mapping_corr_time(corr_time, start_point, time_extent):
    #function remapping in time

    # we are supossing that 1 month is 30 days for using np.timedelta64
    
    
    ### while loop for bigger dates ###
    max_time = time_extent[1]
    time2 = time_extent[0]
    time1 = start_point[0]
    # bearing = 0 donc cos(0)=1 and sin(0)=0
    new_time = [time1]
    while time2 < max_time:
        time2 = time1 + np.timedelta64(corr_time*30,'D')
        new_time.append(time2)
        time1 = time2

        
    ### while loop for smaller dates ###
    min_time = time_extent[0]
    time2 = time_extent[1]
    time1 = start_point[0]
    # bearing = pi donc cos(pi)=-1 and sin(pi)=0
    while time2 > min_time:
        time2 = time1 - np.timedelta64(corr_time*30,'D')
        new_time.append(time2)
        time1 = time2
    
    new_time = np.sort(new_time)
    
    return new_time

    
def BIC_calculation(ds, corr_dist, corr_time, pcm_features, features_in_ds, z_dim, Nrun=10, NK=20):

    #start = time.time()
    #TODO: latitude and longitude values

    # grid extent
    grid_extent = np.array([ds.longitude.values.min(), ds.longitude.values.max(), ds.latitude.values.min(), ds.latitude.values.max()])
    # time extent
    time_extent = np.array([ds.time.values.min(), ds.time.values.max()])

    class_list = np.arange(0,NK) # this is the list of arguments to iterate over, for instance nb of classes for a PCM

    def BIC_cal(X, k):
        """ Function to run on a single argument """
    
        #create model
        m = pcm(K=k+1, features=pcm_features)
        #fit model
        m._classifier.fit(X)
    
        #calculate LOG LIKEHOOD
        llh = m._classifier.score(X)

        # calculate Nb of independant parameters to estimate
        # we suppose m._classifier.covariance_type == 'full'
        _, n_features = m._classifier.means_.shape
        cov_params = m._classifier.n_components * n_features * (n_features + 1) / 2.
        mean_params = n_features * m._classifier.n_components
        Nf = int(cov_params + mean_params + m._classifier.n_components - 1)
    
        #calculate bic
        N_samples = X.shape[0]
        BIC = (-2 * llh * N_samples + Nf * np.log(N_samples))
        #BIC = m._classifier.bic(X)
    
        return BIC, k 

    BIC = np.zeros((NK,Nrun)) 
    #BIC = []
    for run in range(Nrun):
        #print('run=' + str(run))
    
        #random fist point
        latp = np.random.choice(ds.latitude.values, 1, replace=False)
        lonp = np.random.choice(ds.longitude.values, 1, replace=False)
        timep = np.random.choice(ds.time.values, 1, replace=False)
        #mapping
        new_lats, new_lons = mapping_corr_dist(corr_dist=corr_dist, start_point=np.concatenate((lonp,latp)), grid_extent=grid_extent)
        #new_time = mapping_corr_time(corr_time=corr_time, start_point=timep, time_extent=time_extent)
       
    
        #ds_ran = ds.sel(latitude=list(new_lats), longitude=list(new_lons), time=new_time, method='nearest')
        ds_ran = ds.sel(latitude=list(new_lats), longitude=list(new_lons), method='nearest')
        #ds_ran = ds_ran.isel(time=[2,5,8,11])
        ds_ran = ds_ran.isel(time=[5,11])
        # if repeated time values
        #index = np.unique(ds_ran['time'], return_index=True)
        #print(index)
        #ds_ran.isel(time=index[1])
        print(ds_ran.time.values)
    
        # pre-processing
        m = pcm(K=4, features=pcm_features) # K=4 it is not important, it is only used for preproces data
        X , sampling_dims = m.preprocessing(ds_ran, features=features_in_ds, dim=z_dim, action='fit')
    
        #BIC computation in parallel
        results = []
        ConcurrentExecutor = concurrent.futures.ThreadPoolExecutor(max_workers=100)
        with ConcurrentExecutor as executor:
            future_to_url = {executor.submit(BIC_cal, X, k): k for k in class_list}
            futures = concurrent.futures.as_completed(future_to_url)
            futures = tqdm(futures, total=len(class_list))
            for future in futures:
                traj = None
                try:
                    traj = future.result()
                except Exception as e:
                    #pass
                    raise
                finally:
                    results.append(traj)
        results = [r for r in results if r is not None]  # Only keep non-empty results
        results.sort(key=lambda x:x[1])
        BIC[:,run] = np.array([i[0] for i in results])
    
    #end = time.time()
    #print((end - start)/60)
    #calculer minimun et si il y a pas un minimun, warning
    return BIC

def plot_BIC(BIC, NK):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5), dpi=90)
    BICmean = np.mean(BIC,axis=1)
    BICstd = np.std(BIC,axis=1)
    normBICmean = (BICmean-np.mean(BICmean))/np.std(BICmean)
    normBICstd = np.std(normBICmean)
    #plt.plot(np.arange(kmax)+1,(BIC-np.mean(BIC))/np.std(BIC),label='Raw BIC')
    plt.plot(np.arange(NK)+1,BICmean, label='BIC mean')
    plt.plot(np.arange(NK)+1,BICmean+BICstd,color=[0.7]*3,linewidth=0.5, label='BIC std')
    plt.plot(np.arange(NK)+1,BICmean-BICstd,color=[0.7]*3,linewidth=0.5)
    plt.ylabel('BIC')
    plt.xlabel('Number of classes')
    plt.xticks(np.arange(NK)+1)
    plt.legend()
    plt.title('Bayesian information criteria (BIC)')