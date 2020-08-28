import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import xarray as xr

from PIL import Image, ImageFont, ImageDraw

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
        
        # check if dataset should include PCM variables
        assert ("PCM_LABELS" in self.ds), "Dataset should include PCM_LABELS varible to be plotted. Use pyxpcm.predict function with inplace=True option"
        
        # creates dictionary with coordinates
        coords_list = list(self.ds.coords.keys())
        coords_dict = {}
        for c in coords_list:
            axis_at = self.ds[c].attrs.get('axis')
            #print(axis_at)
            if axis_at == 'Y':
                coords_dict.update({'latitude': c})
            if axis_at == 'X':
                coords_dict.update({'longitude': c})
            if axis_at == 'T':
                coords_dict.update({'time': c})
                
        self.coords_dict = coords_dict

        # TODO: assign a data type
        # types: profiles, gridded, timeseries in the future??
        
        # TODO: get information about dataset 
        

        
        
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
        fig.canvas.draw()
        fig.tight_layout()
        plt.margins(0.1)
        
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
                counts_k = pcm_labels_k.groupby(time_variable + '.' + time_bins).count(...)
            else:
                counts_k = xr.concat([counts_k, pcm_labels_k.groupby(time_variable + '.' + time_bins).count(...)], "k")

        
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
        
        
        
    # TODO: def function which adds dataset information to the plot
    
    def add_lowerband(self, mfname, outfname, band_height = 50, color=(255, 255, 255, 255)):
        """ Add lowerband to a figure
    
            Parameters
            ----------
            mfname : string
                source figure file
            outfname : string
                output figure file
        """
        #TODO: do I need to use self here?
        image = Image.open(mfname, 'r')
        image_size = image.size
        width = image_size[0]
        height = image_size[1]
        background = Image.new('RGBA', (width, height + band_height), color)
        background.paste(image, (0, 0))
        background.save(outfname)    
        
        
    def add_2logo(self, mfname, outfname, logo_height=50, txt_color=(0, 0, 0, 255), data_src='CMEMS'):
        """ Add 2 logos and text to a figure
    
            Parameters
            ----------
            mfname : string
                source figure file
            outfname : string
                output figure file
        """
        font_path = "logos/Calibri_Regular.ttf"
        lfname2 = "logos/Blue-cloud_compact_color.png"
        lfname1 = "logos/Logo-LOPS_transparent.png"

        mimage = Image.open(mfname)

        # Open logo images:
        limage1 = Image.open(lfname1)
        limage2 = Image.open(lfname2)

        # Resize logos to match the requested logo_height:
        aspect_ratio = limage1.size[1]/limage1.size[0] # height/width
        simage1 = limage1.resize((int(logo_height/aspect_ratio), logo_height) )

        aspect_ratio = limage2.size[1]/limage2.size[0] # height/width
        simage2 = limage2.resize((int(logo_height/aspect_ratio), logo_height) )

        # Paste logos along the lower white band of the main figure:
        box = (0, mimage.size[1]-logo_height)
        mimage.paste(simage1, box)

        box = (simage1.size[0], mimage.size[1]-logo_height)
        mimage.paste(simage2, box)

        # Add copyright text:
        #txtA = ("© 2017-2019, SOMOVAR Project, %s") % (__author__)
        txtA = "Dataset information:"
        fontA = ImageFont.truetype(font_path, 14)

        txtB = "%s\n Source: %s" % (self.ds.attrs.get('title'), self.ds.attrs.get('credit'))
        fontB = ImageFont.truetype(font_path, 12)

        txtsA = fontA.getsize_multiline(txtA)
        txtsB = fontB.getsize_multiline(txtB)

        xoffset = 5 + simage1.size[0] + simage2.size[0]
        if 0:  # Align text to the top of the band:
            posA = (xoffset, mimage.size[1]-logo_height - 1)
            posB = (xoffset, mimage.size[1]-logo_height + txtsA[1])
        else:  # Align text to the bottom of the band:
            posA = (xoffset, mimage.size[1]-txtsA[1]-txtsB[1]-5)
            posB = (xoffset, mimage.size[1]-txtsB[1]-5)

        # Print
        drawA = ImageDraw.Draw(mimage)
        drawA.text(posA, txtA, txt_color, font=fontA)
        drawB = ImageDraw.Draw(mimage)
        drawB.text(posB, txtB, txt_color, font=fontB)

        # Final save
        mimage.save(outfname)
        
    def save_BlueCloud(self, out_name): #function which saves figure and add logos
        
        #save image
        plt.margins(0.1)
        plt.savefig(out_name, bbox_inches='tight', pad_inches = 0)
        
        #add lower band
        #self.add_lowerband(out_name, out_name, band_height = 120, color=(255, 255, 255, 255))
        self.add_lowerband(out_name, out_name)
        
        #add logo
        #self.add_2logo(out_name, out_name, logo_height=120, txt_color=(0, 0, 0, 255), data_src='CMEMS')
        self.add_2logo(out_name, out_name)
        
        

        
    pass