import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import plotly.express as px
import cartopy.crs as ccrs
import skytools
import jmtools as jt
from jmtools import geodetic2aer
from jmtools import interferometry
import cartopy.feature as cfeature
from scipy.interpolate import BarycentricInterpolator

class Data_Package:
    def __init__(self, df,tx,rx):
        self.df=df
        self.tx_latlon=tx
        self.rx_latlon=rx
        self.wavelegnth=.003

    
    def plotFlightPath(self):

        #Get time since initial recordings
        flightmin=min(self.df['time']) 
        flighttime=self.df['time']-flightmin

        fig = plt.figure(dpi=800)#figsize=(20, 20), layout='constrained')
        #fig.suptitle("Flight Path")

        gs1 = GridSpec(15,15, left=0.05, right=0.98, wspace=0.05)

        # Min and max based on colorado Coordinates
        max_plot_lat=-104
        min_plot_lat=-106
        max_plot_lon=40.3
        min_plot_lon=39.2

        #Plot flight over continent with tx and rx relative locations
        #TODO: Write program to automatically determine latitude/longigude displayed when graphing
        tx_rx_plot = plt.subplot(gs1[0:10,:10], projection=ccrs.PlateCarree())
        tx_rx_plot.coastlines()
        tx_rx_plot.set_extent([min_plot_lat, max_plot_lat, min_plot_lon, max_plot_lon], crs=ccrs.PlateCarree())
        tx_rx_plot.scatter(self.df['lat'], self.df['lon'], color='indigo', linewidth=2, marker='o')
        tx_rx_plot.scatter(self.tx_latlon[0],self.tx_latlon[1],color='red') 
        tx_rx_plot.scatter(self.rx_latlon[0],self.rx_latlon[1],color='blue')
        tx_rx_plot.set_title("Flight vs Transmitter/Receiver")

        #Plot flight on world map
        worldplot=plt.subplot(gs1[0:3, 11:], projection=ccrs.PlateCarree())
        worldplot.coastlines()
        worldplot.add_feature(cfeature.BORDERS)
        worldplot.set_title("Flight on PlateCarree")
        worldplot.scatter(self.df['lat'], self.df['lon'], color='blue', linewidth=2, marker='o')
        worldplot.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        
        #plot flight location vs time
        pathplot=plt.subplot(gs1[4:8,11:], projection=ccrs.PlateCarree())
        pathplot.set_title("Flight Location vs Time")
        pathplot.coastlines()
        pathplot.set_extent([min_plot_lat, max_plot_lat, min_plot_lon, max_plot_lon], crs=ccrs.PlateCarree()) 
        pathplot.scatter(self.df['lat'], self.df['lon'], c=flighttime, cmap='cividis') #I don't know why I need this line and the one bellow but I do
        scatter = pathplot.scatter(self.df['lon'], self.df['lat'], c=flighttime, cmap='cividis')
        cbar = plt.colorbar(scatter, ax=pathplot, orientation='vertical')
        cbar.set_label('Time (s)')  # Set the label for the colorbar

        #Plot altitide as function os time
        altplot = fig.add_subplot(gs1[11:, 0:5])
        altplot.scatter(flighttime, self.df['geoaltitude']) #label Axies, Title where it's taking off from (Google Callsign)
        altplot.set_title("Flight Altitude vs Time")
        altplot.set_xlabel("Time (s)")
        altplot.set_ylabel("Altitude (m)")

        #plot latitude vs time
        lat_time_plot=fig.add_subplot(gs1[11:,6:10])
        lat_time_plot.scatter(flighttime,self.df['lat'])
        lat_time_plot.set_title("Flight Latitude vs Time")
        lat_time_plot.set_xlabel("Time (s)")
        lat_time_plot.set_ylabel("Latitude")

        #Plot longitude vs time
        lat_time_plot=fig.add_subplot(gs1[11:,11:])
        lat_time_plot.scatter(flighttime,self.df['lon'])
        lat_time_plot.set_title("Flight Longitude vs Time")
        lat_time_plot.set_xlabel("Time (s)")
        lat_time_plot.set_ylabel("Longitude")

        plt.show()
    def name_thing(self):
        self.word="word"

    #Convert Coordinates
    def convertCoords(self):
        earth_measurements=[6378,6356] #Earth Measurements
        #TODO: I need to verify these numbers as I currentley don't seem to be able to access the gitlab
        self.tx_xyz=[0,0,0]
        self.rx_xyz=[0,0,0]
        self.df['X'],self.df['Y'],self.df['Z']=jt.geodetic2aer.geodetic2ecef(self.df['lat'],self.df['lon'],.001*self.df['geoaltitude'],earth_measurements[0],earth_measurements[1])
        self.tx_xyz[0],self.tx_xyz[1],self.tx_xyz[2]=jt.geodetic2aer.geodetic2ecef(self.tx_latlon[0],self.tx_latlon[1],.001*self.tx_latlon[2],earth_measurements[0],earth_measurements[1])
        self.rx_xyz[0],self.rx_xyz[1],self.rx_xyz[2]=jt.geodetic2aer.geodetic2ecef(self.rx_latlon[0],self.rx_latlon[1],.001*self.rx_latlon[2],earth_measurements[0],earth_measurements[1])
        #return df,tx_xyz,rx_xyz
    
    def bistaticrange(self,plot):
        x=self.df['X']
        y=self.df['Y']
        z=self.df['Z']
        tx_range=np.sqrt((self.tx_xyz[0]-x)**2+(self.tx_xyz[1]-y)**2+(self.tx_xyz[2]-z)**2) #Calculate distance from plane to tx
        rx_range=np.sqrt((self.rx_xyz[0]-x)**2+(self.rx_xyz[1]-y)**2+(self.rx_xyz[2]-z)**2) #Calculate distance from plane to rx
        bistatic_range=tx_range+rx_range #Sum to return bistatic range
        self.df['brange']=bistatic_range

        if plot==True:
            #Plot bistatic range vs time
            plt.scatter(self.df['time'],self.df['brange'])
            plt.title("bistatic range vs time")
            plt.xlabel("Time (Unix Timestamp)")
            plt.ylabel("bistatic range (km)")
    
    def interpolate(self,timeseries): #timeseries is the set of times that the data should be interpolated on.
        P=BarycentricInterpolator(self.df['time'],self.df['brange'])#Creates a function P which values can be put into
        brange_int=P(timeseries)
        self.df_intermolated=pd.DataFrame({'time':timeseries,'brange':brange_int})