import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xarray as xr

class Plotter:
    '''New class for visualisation of data from pyxpcm
    
       Parameters
       ----------
           ds: dataset including PCM results
           m: pyxpcm model
           data_type: data type
           
           '''        
        
    def __init__(self, ds, m, data_type):
        
        self.ds = ds
        self.m = m
        self.data_type = data_type
        # check coordinates in dataset and assign a data type
        
        # types: profiles, gridded, timeseries in the future??
        # look for latitude and longitude variables inside functions?
        
        # TODO: get information about dataset 
        # ds['latitude'].attrs
        #l = ds.coords
        #l.keys
        #ds.indexes.keys
        
        # check if dataset should include PCM variables
        assert ("PCM_LABELS" in self.ds), "Dataset should include PCM_LABELS varible to be plotted. Use pyxpcm.predict function with inplace=True option"
        
        
    def vertical_structure(self,q_variable):
        '''Plot vertical structure of each class
        
           Parameters
           ----------
               q_variable: quantile variable calculated with pyxpcm.quantile function (inplace=True option)
               
           Returns
           -------
           
               '''
        
        # TODO: check if data is profile: difference between profiles, gridded profiles and gridded
        # TODO: try with other k values
        # TODO: Plot function already in pyxpcm
        fig, ax = self.m.plot.quantile(self.ds[q_variable], maxcols=4, figsize=(10, 8), sharey=True)
        # TODO: add dataset information (function)
        
    def spatial_distribution(self, proj, extent, co):
        '''Plot spatial distribution of classes
        
           Parameters
           ----------
               proj: projection
               extent: map extent
               co: coordinates names co={'longitude':'LONGITUDE', 'latitude':'LATITUDE'}
               Input dataset should have only one time step
               
           Returns
           -------
           
               '''
        
        #TODO: if finally we use k-means instead of GMM with huge datsets, we can not plot posteriors
        
        subplot_kw={'projection': proj, 'extent': extent}
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), dpi=120, facecolor='w', edgecolor='k', subplot_kw=subplot_kw)
        kmap = self.m.plot.cmap() # TODO: function already in pyxpcm
        
        # check if gridded or profiles data
        if self.data_type == 'profiles':
            sc = ax.scatter(self.ds[co['longitude']], self.ds[co['latitude']], s=3, c=self.ds['PCM_LABELS'], cmap=kmap, transform=proj, vmin=0, vmax=self.m.K)
        if self.data_type == 'gridded':
            sc = ax.pcolormesh(self.ds[co['longitude']], self.ds[co['latitude']], self.ds['PCM_LABELS'], 
                               cmap=kmap, transform=proj, vmin=0, vmax=self.m.K)
        
        cl = self.m.plot.colorbar(ax=ax) # TODO: function already in pyxpcm
        gl = self.m.plot.latlongrid(ax, dx=10) # TODO: function already in pyxpcm
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        ax.set_title('LABELS of the training set')
        
        # TODO: add dataset information (function)
        
        #plt.show()
        
        
    def plot_posteriors(self, proj, extent, co):
        '''Plot posteriors in a map
        
           Parameters
           ----------
               proj: projection
               extent: map extent
               co: coordinates names co={'longitude':'LONGITUDE', 'latitude':'LATITUDE'}
               Input dataset should have only one time step
           
           Returns
           -------
           
           '''
        
        # check if PCM_POST variable exists
        assert ("PCM_POST" in self.ds), "Dataset should include PCM_POST varible to be plotted. Use pyxpcm.predict_proba function with inplace=True option"
        
        cmap = sns.light_palette("blue", as_cmap=True)
        subplot_kw={'projection': proj, 'extent': extent}
        fig, ax = self.m.plot.subplots(figsize=(10,22), maxcols=2, subplot_kw=subplot_kw)# TODO: function already in pyxpcm

        for k in self.m:
            if self.data_type == 'profiles':
                sc = ax[k].scatter(self.ds[co['longitude']], self.ds[co['latitude']], s=3, c=self.ds['PCM_POST'].sel(pcm_class=k),
                           cmap=cmap, transform=proj, vmin=0, vmax=1)
            if self.data_type == 'gridded':
                sc = ax[k].pcolormesh(self.ds[co['longitude']], self.ds[co['latitude']], self.ds['PCM_POST'].sel(pcm_class=k),
                                      cmap=cmap, transform=proj, vmin=0, vmax=1)
                    
            
            cl = plt.colorbar(sc, ax=ax[k], fraction=0.03)
            gl = self.m.plot.latlongrid(ax[k], fontsize=8, dx=20, dy=10)
        
            ax[k].add_feature(cfeature.LAND)
            ax[k].add_feature(cfeature.COASTLINE)
            ax[k].set_title('PCM Posteriors for k=%i' % k)
            
            
        fig.suptitle(r"$\bf{"'PCM  Posteriors'"}$" + ' \n (probability of a profile to belong to a class k)')
        fig.canvas.draw()
        fig.tight_layout()
        #fig.subplots_adjust(top=0.95)
            
           # TODO: add dataset information (function)

        
    def temporal_distribution(self, time_variable, time_bins, pond):
        '''Plot temporal distribution of classes by moth or by season
        
           Parameters
           ----------
               time_variable: time variable name
               time_bins: 'month' or 'season'
               pond: 'abs' or 'rel' (divided by total nomber of observation in time bin)
            
            Returns
            -------
            
        '''
        
        
        # check if more than one temporal step
        assert (len(self.ds[time_variable]) > 1), "Length of time variable should be > 1"
                        
        # data to be plot
        # TODO: is it the best way??
        pcm_labels = self.ds['PCM_LABELS']
       
        if time_bins == 'month':
            xaxis_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        if time_bins == 'season':
            seasons_dict = {1: 'DJF', 2: 'MAM', 3: 'JJA', 4:'SON'}
            xaxis_labels = ['DJF', 'MAM', 'JJA', 'SON']
        
        width = 0.8/(self.m.K)  # the width of the bars
        fig, ax = plt.subplots(figsize=(18,10))
        kmap = self.m.plot.cmap() # TODO: function already in pyxpcm
        
        #loop in k for counting 
        for cl in range(self.m.K):
            #get time array with k=cl
            pcm_labels_k = pcm_labels.where(pcm_labels == cl)
                       
            if cl == 0:
                #counts_k = pcm_labels_k.groupby(time_variable + '.' + time_bins).count()
                counts_k = pcm_labels_k.groupby(time_variable + '.' + time_bins).count(...)
            else:
                #counts_k = xr.concat([counts_k, pcm_labels_k.groupby(time_variable + '.' + time_bins).count()], "k")
                counts_k = xr.concat([counts_k, pcm_labels_k.groupby(time_variable + '.' + time_bins).count(...)], "k")
            #print(counts_k)
        #print(sum(counts_k))
        
        if pond == 'rel':
            counts_k = counts_k/sum(counts_k)
            
        #loop for plotting
        for cl in range(self.m.K): 
            
            if time_bins == 'month':
                bar_plot_k = ax.bar(counts_k.month - (self.m.K/2-cl)*width, counts_k.isel(k=cl), width, label = 'K =' + str(cl))
            if time_bins == 'season':
                x_ticks_k = []
                for i in range(len(counts_k.season)):
                    x_ticks_k.append(list(seasons_dict.values()).index(counts_k.season[i])+1)
                    # print(x_ticks_k)
                #plot
                bar_plot_k = ax.bar(np.array(x_ticks_k) - (self.m.K/2-cl)*width, counts_k.isel(k=cl), width, label = 'K =' + str(cl))
                #cmap=kmap
            
    
        # format
        title_string = r'Number of profiles in each class by $\bf{' + time_bins + '}$'
        ylabel_string = 'Number of profiles'
        if pond == 'rel':
            title_string = title_string + '\n (divided by total number of profiles in each bin)'
            ylabel_string = 'Relative number of profiles'
        
        
        ax.set_xticks(np.arange(1,len(xaxis_labels)+1))
        ax.set_xticklabels(xaxis_labels, fontsize=12)
        plt.yticks(fontsize=12)
        ax.legend(fontsize=12)
        ax.set_ylabel(ylabel_string,fontsize=12)
        ax.set_title(title_string, fontsize=14)
        fig.tight_layout()
        # TODO: same colors for each class in all figures

        # TODO: add dataset information (function)
        

        #plt.show()
        
        
        
    # def function which adds dataset information to the plot
        
    #def save_figure(self, time_variable, time_bins, pond): #function which saves figure a add logos
        # plt.savefig('ArgoMed_months_hist_EX.png')
        
    pass