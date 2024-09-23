# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:09:59 2024

@author: orlaj
"""
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import nearest_points

class check_weather:
    def __init__(self,senario):    
        
        ##read in CMIP6 data
        #historical data
        self.hist_data_ts=xr.open_dataset(r"CMIP6 data\Hist\downscalled_CMIP6_hist_ts.nc")
        self.hist_data_wind=xr.open_dataset(r"CMIP6 data\Hist\downscalled_CMIP6_wind_hist.nc")
        self.hist_data_pr=xr.open_dataset(r"CMIP6 data\Hist\downscalled_CMIP6_hist_pr_divided_by_4.nc")
        self.ocean_temp_hist=xr.open_dataset(r"CMIP6 data\Hist/ts_Amon_MRI-ESM2-0_historical_r1i1p1f1_gn_185001-201412.nc",engine='netcdf4')

        #senario data
        self.senario=str(senario)
        self.pr=xr.open_dataset(r"CMIP6 data\SSPs/"+self.senario+'/downscalled_CMIP6_SSP'+self.senario+'_pr_divided_by_4.nc')
        self.ts=xr.open_dataset(r"MIP6 data\SSPs/"+self.senario+'/downscalled_CMIP6_SSP'+self.senario+'_ts.nc')
        self.wind=xr.open_dataset(r"CMIP6 data\SSPs/"+self.senario+'/downscalled_CMIP6_SSP'+self.senario+'_wind.nc')
        self.ocean_temp=xr.open_dataset(r"CMIP6 data\SSPs/"+self.senario+'/ts_Amon_MRI-ESM2-0_ssp'+self.senario+'_r1i1p1f1_gn_201501-210012.nc',engine='netcdf4')
        
        
        #patching together the historical and senario so that the simulations can start in 2011
        end_date = self.ocean_temp_hist.time.max().values
        start_date = end_date - np.timedelta64(4*365, 'D')
        last_years = self.ocean_temp_hist.sel(time=slice(start_date, end_date))
        self.ocean_temp = xr.concat([last_years, self.ocean_temp], dim='time')
        self.ocean_temp=self.ocean_temp.sel(lat=slice(*[10, 15]), lon=slice(*[80, 93.5]))

        end_date = self.hist_data_ts.time.max().values
        start_date = end_date - np.timedelta64(4*365, 'D')
        last_years = self.hist_data_ts.sel(time=slice(start_date, end_date))
        self.ts = xr.concat([last_years, self.ts], dim='time')

        last_years = self.hist_data_wind.sel(time=slice(start_date, end_date))
        self.wind = xr.concat([last_years, self.wind], dim='time')

        last_years = self.hist_data_pr.sel(time=slice(start_date, end_date))
        self.pr = xr.concat([last_years, self.pr], dim='time')
        
        #getting coordinate data
        self.lat = self.pr.lat
        self.lon = self.pr.lon
        
        #reading in upazila shapefile
        bgd_shapefile_path = r"shapefiles\gadm41_BGD_3.shp"
        self.gdf = gpd.read_file(bgd_shapefile_path)
        self.gdf=self.gdf.drop(209)
        self.gdf=self.gdf.sort_values(by='CC_3')
        
    def binary_checker(self):
        #an early weather checking function, gives 0 if there is no weather event and 1 if there is

        #calculate the likelihood of weather events
        F=self.flood_index()
        C=self.cyclone_finder()
        H=self.heatwave_finder()
        
        #if flood indicaor is less than 0, give it 0, otherwise 1
        binary_F = xr.where(F['I_flood'] < 0.2, 0, 1)
        self.F_per_upazila=self.region_assign(binary_F,F=True)
        
        #if there is a value after cyclone masking and the windspeed is high enough for a cyclone, give 1
        mask = (~np.isnan(C['wind_speed'])) & (C['wind_speed'] > 8)
        binary_C = xr.where(mask, 1, 0)
        self.C_per_upazila=self.region_assign(binary_C,F=False)
        
        #if there is a value after heatwave masking, give 1
        binary_H = xr.where(np.isnan(H['ts']), 0, 1)
        self.H_per_upazila=self.region_assign(binary_H,F=False)
        
        return self.F_per_upazila,self.C_per_upazila,self.H_per_upazila
    
    def normalised_checker(self):
        #the updated weather checking function, where insensity is scaled and normalised

        #calculate the likelihood of weather events
        F=self.flood_index()
        C=self.cyclone_finder()
        H=self.heatwave_finder()
        
        #find where there is enough risk of flooding and keep only those values
        F['I_flood'] = xr.where(F['I_flood'] > 0.2, F['I_flood'], 0)
        #assign values to upazilas
        self.F_per_upazila=self.region_assign(F,F=True)
        #normalise the flood index
        min_val = self.F_per_upazila['I_flood'].min()
        max_val = self.F_per_upazila['I_flood'].max()
        self.F_per_upazila['I_flood'] = (self.F_per_upazila['I_flood'] - min_val) / (max_val - min_val)
        
        #find where there is enough risk of cyclones and keep only those values
        C['wind_speed'] =xr.where((~np.isnan(C['wind_speed'])) & (C['wind_speed'] > 8),C['wind_speed'],0)
        #normalise the windspeeeds to make an index
        min_val = C['wind_speed'].min()
        max_val = C['wind_speed'].max()
        C['wind_speed'] = (C['wind_speed'] - min_val) / (max_val - min_val)
        #assign values to upazilas
        self.C_per_upazila=self.region_assign(C,F=False)
        
        #normalise heatwave temperatures to make an index
        min_val = H['ts'].min()
        max_val = H['ts'].max()
        H['ts'] = (H['ts'] - min_val) / (max_val - min_val)
        #assign values to upazilas
        self.H_per_upazila=self.region_assign(H,F=False)
        
        return self.F_per_upazila,self.C_per_upazila,self.H_per_upazila
            
        
    def flood_index(self):
        #a flood risk index based on the calculations of Deo et al., 2018

        #lists to hold effective percipitation values
        P_E=[]
        max_P_E=[]

        #getting time values
        index_time=self.pr.time
        years = np.unique(self.pr['time.year'])

        #calculating effective percipitation
        for k in range(12): #for first year where we cannot calculate the index
            P_E.append(self.pr.pr[k]*0) #assign 0's for the year
        for i in range(12,len(index_time)):
            #monthly version of equation from Deo et al., 2018
            P_E.append(24*(self.pr.pr[i]+0.39*self.pr.pr[i-1]+0.28*self.pr.pr[i-2]+0.22*self.pr.pr[i-3]+0.17*self.pr.pr[i-4]+0.14*self.pr.pr[i-5]+0.11*self.pr.pr[i-6]+0.09*self.pr.pr[i-7]+0.07*self.pr.pr[i-8]+0.05*self.pr.pr[i-9]+0.03*self.pr.pr[i-10]+0.02*self.pr.pr[i-11]))
            
        #Create an xarray dataset for P_E
        P_E_data = np.array(P_E)
        P_E_xr = xr.DataArray(P_E_data, coords=[index_time, self.lat, self.lon], dims=['time', 'lat', 'lon'])
        P_E_dataset = xr.Dataset({'P_E': P_E_xr})
        
        #find the maximum effective percipitation from the time period
        for year in years:
            for lat_val in self.lat:
                for lon_val in self.lon:
                    P_E_yearly_grid = P_E_dataset.sel(time=str(year), lat=lat_val, lon=lon_val)
                    max_P_E.append(P_E_yearly_grid) 
        
        max_P_E_array = xr.concat(max_P_E, dim='year')  # Concatenate along 'year' dimension
        mean_max_P_E = max_P_E_array.mean(dim='year')  # Calculate mean over years
        
        std_max_P_E = max_P_E_array.std(dim='year')  # Calculate standard deviation over years
        
        #calculate the flood index
        I_flood = (P_E_dataset - mean_max_P_E) / std_max_P_E
        I_flood = I_flood.rename({'P_E': 'I_flood'})
        
        return I_flood
    
    def cyclone_finder(self):
        #thresholds found from historic data
        wind_threshold=8.1356349#m/s
        temp_threshold=303.343885#K
        
        #masking datat to only keep what is above the cyclone threshold
        cyclone_mask = (
            (self.wind.wind_speed.isel(plev=0).values > wind_threshold).any() &
            (self.ocean_temp.ts.mean(dim=['lat','lon']) > temp_threshold)
        )
        cyclone_wind = xr.where(cyclone_mask.astype("float64"), self.wind.isel(plev=0).astype("float64"), 0)

        return cyclone_wind
    
    def heatwave_finder(self):
        #finding times of heatwaves based on the definition by Nissan et al., 2017

        #find the 95th percentile temperature from the historical data
        p95=self.hist_data_ts.quantile(0.95)
        
        #filter to keep values >= 95th percentile
        heatwave = self.ts.where(self.ts >= p95, drop=True)

        return heatwave
    
    def region_assign(self,data,F=False):
        #function for assigning data to upazilas

        #list for holding data
        data_at_centroid = []
        
        #finding distance to water bodies for flood index calculation
        if F:
            #list for holding distances
            river_dist=[]

            #load in rivers and lakes shapefile
            river_shapefile_path = r"shapefiles\bgd_watcrsa_1m_iscgm.shp"
            gdf_river = gpd.read_file(river_shapefile_path)

            for index, region in self.gdf.iterrows(): #run through each upazila
                
                #check if any river intersects with the current region
                intersecting_rivers = gdf_river[gdf_river.intersects(region.geometry)]
                if not intersecting_rivers.empty:
                    #if there is rivers or lakes in the given upazila, append 1
                    river_dist.append(1)

                else:
                    #otherwise, fins the distance to the nearest waterbody
                    nearest_river_distances = []

                    if isinstance(region.geometry, Polygon): #had two different types of polygons in the shapefile for some reason
                        polygon = region.geometry
                        nearest_river = nearest_points(polygon.centroid, gdf_river.unary_union)[1]
                        distance_to_nearest_river = polygon.centroid.distance(nearest_river)
                        nearest_river_distances.append(distance_to_nearest_river)
                    if isinstance(region.geometry, MultiPolygon):
                        for polygon in region.geometry.geoms:
                            nearest_river = nearest_points(polygon.centroid, gdf_river.unary_union)[1]
                            distance_to_nearest_river = polygon.centroid.distance(nearest_river)
                            nearest_river_distances.append(distance_to_nearest_river)
                    distance_to_nearest_river = min(nearest_river_distances)

                    #take 1 minus that distance so that it can be used to scale the flood index values
                    river_dist.append(1-distance_to_nearest_river)

            #add this distance factor to the upazila data
            self.gdf['river_dist'] = river_dist

        #loop through each upazila
        for index, row in self.gdf.iterrows():
            #find central coordinates of the upazila
            centroid = row['geometry'].centroid
            centroid_lat = centroid.y
            centroid_lon = centroid.x
            
            #find the data at the central coorinate point
            if F:
                #if this is floods being calculated, include the river distance factor
                data_at_centroid.append(data.sel(lat=centroid_lat, lon=centroid_lon, method='nearest')*row['river_dist'])

            else:
                data_at_centroid.append(data.sel(lat=centroid_lat, lon=centroid_lon, method='nearest'))
            
        #create an xarray for the data
        combined_data = xr.concat(data_at_centroid, dim='region')

        #let the upazila ID act as a coordinate for the data
        combined_data = combined_data.assign_coords(region=self.gdf.loc[:,'CC_3'].astype(int))
    
        return combined_data
    
    def plot_upazilas(self,t):
        #a function to plot the weather event data for quick checking

        #find the weather events
        F_plot,C_plot,H_plot=self.binary_checker()

        #set up for running over each weather type
        variables=[F_plot,C_plot,H_plot]
        title=['Flood warning ','Cyclone warning ','Heatwave warning']
        counter=0

        for V in variables:
            #find data at given time
            C_values = V.isel(time=t) 

            #set up colour map
            cmap = plt.cm.RdYlGn_r
            min_value = np.min(C_values)
            max_value = np.max(C_values)
            norm = Normalize(vmin=min_value, vmax=max_value)
            
            #set up plot
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
            ax.coastlines(resolution='50m')
            ax.add_feature(cfeature.LAND)
            
            #set up plot bounds
            min_lon = np.inf
            max_lon = -np.inf
            min_lat = np.inf
            max_lat = -np.inf
            
            #run through each upazila and plot its data
            for (idx, (index, row)) in enumerate(self.gdf.iterrows()):
                color = cmap(norm(C_values[idx]))
                ax.add_geometries([row['geometry']], ccrs.PlateCarree(), facecolor=color, edgecolor='black')
            
                #update plot bounds
                min_lon = min(min_lon, row['geometry'].bounds[0])
                max_lon = max(max_lon, row['geometry'].bounds[2])
                min_lat = min(min_lat, row['geometry'].bounds[1])
                max_lat = max(max_lat, row['geometry'].bounds[3])
            
            #set plot bounds
            ax.set_extent([min_lon, max_lon, min_lat, max_lat])

            #define title
            timestamp = pd.Timestamp(V.time.isel(time=t).values)
            formatted_date = timestamp.strftime('%B %Y')
            ax.set_title(title[counter]+formatted_date)

            #update counter
            counter+=1

            #define colour bar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([]) 
            plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.05)
            plt.subplots_adjust(right=0.85)
            
            plt.show()