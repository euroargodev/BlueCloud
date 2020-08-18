import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Plotter:
    '''New class for visualisation of data from pyxpcm
        ds: dataset including PCM results'''        
        
    def __init__(self, ds, m, data_type):
        
        self.ds = ds
        self.m = m
        self.data_type = data_type
        # types: profiles, gridded, timeseries in the future??
        # look for latitude and longitude variables inside functions?
        
        # check if dataset should include PCM variables
        assert ("PCM_LABELS" in self.ds), "Dataset should include PCM_LABELS varible to be plotted. Use pyxpcm.predict function with inplace=True option"
        
        
    def vertical_structure(self,q_variable):
        '''Plot vertical structure of each class
           q_variable: quantile variable calculated with pyxpcm.quantile function (inplace=True option)'''
        
        # TODO: check if data is profile: difference between profiles, gridded profiles and gridded
        # TODO: try with other k values
        # TODO: Plot function already in pyxpcm
        fig, ax = self.m.plot.quantile(self.ds[q_variable], maxcols=4, figsize=(10, 8), sharey=True)
        
        
    def spatial_distribution(self, proj, extent, co):
        '''Plot spatial distribution of classes
           proj: projection
           extent: map extent
           co: coordinates names co={'longitude':'LONGITUDE', 'latitude':'LATITUDE'}'''
              
        subplot_kw={'projection': proj, 'extent': extent}
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), dpi=120, facecolor='w', edgecolor='k', subplot_kw=subplot_kw)
        kmap = self.m.plot.cmap() # TODO: function already in pyxpcm
        
        # check if gridded or profiles data
        if self.data_type == 'profiles':
            sc = ax.scatter(self.ds[co['longitude']], self.ds[co['latitude']], s=3, c=self.ds['PCM_LABELS'], cmap=kmap, transform=proj, vmin=0, vmax=self.m.K)
        if self.data_type == 'gridded':
            # TODO: if time series make subplots
            # for t in np.arange(len(self.ds.time)):
            sc = ax.pcolormesh(self.ds[co['longitude']], self.ds[co['latitude']], self.ds['PCM_LABELS'].isel(time=0), 
                               cmap=kmap, transform=proj, vmin=0, vmax=self.m.K)
        
        cl = self.m.plot.colorbar(ax=ax) # TODO: function already in pyxpcm
        gl = self.m.plot.latlongrid(ax, dx=10) # TODO: function already in pyxpcm
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        ax.set_title('LABELS of the training set')
        
        # saving figure outside??
        #plt.savefig('ArgoMed_map_labels_EX.png')
        
        plt.show()
        
        
    def plot_posteriors(self, proj, extent, co):
        '''Plot posteriors in a map
           proj: projection
           extent: map extent
           co: coordinates names co={'longitude':'LONGITUDE', 'latitude':'LATITUDE'}'''
        
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
                # TODO: if time series 
                # for t in np.arange(len(ds.time)):
                sc = ax[k].pcolormesh(self.ds[co['longitude']], self.ds[co['latitude']], self.ds['PCM_POST'].isel(time=0, pcm_class=k),
                                      cmap=cmap, transform=proj, vmin=0, vmax=1)
                    
            
            cl = plt.colorbar(sc, ax=ax[k], fraction=0.03)
            gl = self.m.plot.latlongrid(ax[k], fontsize=8, dx=20, dy=10)
        
            ax[k].add_feature(cfeature.LAND)
            ax[k].add_feature(cfeature.COASTLINE)
            ax[k].set_title('PCM Posteriors for k=%i' % k)
            
            # saving figure outside??
            #plt.savefig('ArgoMed_posteriors_EX.png')

        
    def temporal_distribution(self, time_variable, time_bins, pond):
        '''Plot temmporal distribution of classes by moth or by season
           time_variable: time variable name
           time_bins: 'month' or 'season'
           pond: 'abs' or 'rel' (divided by total nomber of observation in time bin)'''
        
        # TODO: it only works for profile type data for the moment (how to do when gridded?)
        
        # check if more than one temporal step
        assert (len(self.ds[time_variable]) > 1), "Length of time variable should be > 1"
                        
        # data to be plot
        # TODO: is it the best way??
        pcm_labels = self.ds['PCM_LABELS']
       
        if time_bins == 'month':
            xaxis_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dic']
        if time_bins == 'season':
            seasons_dict = {1: 'DJF', 2: 'MAM', 3: 'JJA', 4:'SON'}
            xaxis_labels = ['DJF', 'MAM', 'JJA', 'SON']
        
        width = 0.5/(self.m.K)  # the width of the bars
        fig, ax = plt.subplots(figsize=(18,10))

        #loop in k and counting in months
        for cl in range(self.m.K):
            #get time array with k=cl
            pcm_labels_k = pcm_labels[pcm_labels == cl]
            # count for each month
            counts_k = pcm_labels_k.groupby(time_variable + '.' + time_bins).count()
            
            if pond == 'rel':
                counts_k = counts_k/sum(counts_k)
            
            if time_bins == 'month':
                bar_plot_k = ax.bar(counts_k.month - (self.m.K/2-cl)*width, counts_k, width, label = ['K =' + str(cl)])
            if time_bins == 'season':
                x_ticks_k = []
                for i in range(len(counts_k.season)):
                    x_ticks_k.append(list(seasons_dict.values()).index(counts_k.season[i])+1)
                    # print(x_ticks_k)
                #plot
                bar_plot_k = ax.bar(np.array(x_ticks_k) - (self.m.K/2-cl)*width, counts_k, width, label = ['K =' + str(cl)])
            
    
        # format
        # TODO: titles when pond = 'rel'
        ax.set_xticks(np.arange(1,len(xaxis_labels)+1))
        ax.set_xticklabels(xaxis_labels, fontsize=12)
        plt.yticks(fontsize=12)
        ax.legend(fontsize=12)
        ax.set_ylabel('Number of profiles',fontsize=12)
        ax.set_title('Number of profiles for each class by month', fontsize=14)
        fig.tight_layout()

        # saving figure outside??
        # plt.savefig('ArgoMed_months_hist_EX.png')

        plt.show()
        
    pass