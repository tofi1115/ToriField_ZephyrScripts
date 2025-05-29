#All of the code for writing a function or set of functions that takes tx and rx and does everything necesary to get
#A phase wraping from them
#By Tori Field

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

print("Reading Single_Flight_tools")

#Function to plot flight possision vs time aspects.

def plotFlightPath(flightdf,tx,rx):

    #Get time since initial recordings
    flightmin=min(flightdf['time']) 
    flighttime=flightdf['time']-flightmin

    fig = plt.figure(dpi=800)#figsize=(20, 20), layout='constrained')
    #fig.suptitle("Flight Path")

    gs1 = GridSpec(15,15, left=0.05, right=0.98, wspace=0.05)

    #Determine latitude/longitude ranges to be displayed
    max_lat=max(flightdf['lat'])
    min_lat=min(flightdf['lat'])
    max_lon=max(flightdf['lon'])
    min_lon=min(flightdf['lon'])
    max_plot_lat=max(tx[0],rx[0],max_lat)+5
    min_plot_lat=min(tx[0],rx[0],min_lat)-5
    max_plot_lon=max(tx[1],rx[1],max_lon)+5
    min_plot_lon=min(tx[1],rx[1],min_lon)-5

    #Plot flight over continent with tx and rx relative locations
    #TODO: Write program to automatically determine latitude/longigude displayed when graphing
    tx_rx_plot = plt.subplot(gs1[0:10,:10],projection=ccrs.PlateCarree())
    tx_rx_plot.coastlines()
    tx_rx_plot.set_extent([min_plot_lat, max_plot_lat, min_plot_lon, max_plot_lon], crs=ccrs.PlateCarree())
    tx_rx_plot.scatter(flightdf['lat'], flightdf['lon'], color='indigo', linewidth=2, marker='o')
    tx_rx_plot.scatter(tx[0],tx[1],color='red') 
    tx_rx_plot.scatter(rx[0],rx[1],color='blue')
    tx_rx_plot.set_title("Flight vs Transmitter/Receiver")

    #Plot flight on world map
    worldplot=plt.subplot(gs1[0:3, 11:], projection=ccrs.PlateCarree())
    worldplot.coastlines()
    worldplot.add_feature(cfeature.BORDERS)
    worldplot.set_title("Flight on PlateCarree")
    worldplot.scatter(flightdf['lat'], flightdf['lon'], color='blue', linewidth=2, marker='o')
    worldplot.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    
    #plot flight location vs time
    pathplot=plt.subplot(gs1[4:8,11:], projection=ccrs.PlateCarree())
    pathplot.set_title("Flight Location vs Time")
    pathplot.coastlines()
    pathplot.set_extent([min_plot_lat, max_plot_lat, min_plot_lon, max_plot_lon], crs=ccrs.PlateCarree()) 
    pathplot.scatter(flightdf['lat'], flightdf['lon'], c=flighttime, cmap='cividis') #I don't know why I need this line and the one bellow but I do
    scatter = pathplot.scatter(flightdf['lon'], flightdf['lat'], c=flighttime, cmap='cividis')
    cbar = plt.colorbar(scatter, ax=pathplot, orientation='vertical')
    cbar.set_label('Time (s)')  # Set the label for the colorbar

    #Plot altitide as function os time
    altplot = fig.add_subplot(gs1[11:, 0:5])
    altplot.scatter(flighttime, flightdf['geoaltitude']) #label Axies, Title where it's taking off from (Google Callsign)
    altplot.set_title("Flight Altitude vs Time")
    altplot.set_xlabel("Time (s)")
    altplot.set_ylabel("Altitude (m)")

    #plot latitude vs time
    lat_time_plot=fig.add_subplot(gs1[11:,6:10])
    lat_time_plot.scatter(flighttime,flightdf['lat'])
    lat_time_plot.set_title("Flight Latitude vs Time")
    lat_time_plot.set_xlabel("Time (s)")
    lat_time_plot.set_ylabel("Latitude")

    #Plot longitude vs time
    lat_time_plot=fig.add_subplot(gs1[11:,11:])
    lat_time_plot.scatter(flighttime,flightdf['lon'])
    lat_time_plot.set_title("Flight Longitude vs Time")
    lat_time_plot.set_xlabel("Time (s)")
    lat_time_plot.set_ylabel("Longitude")

    plt.show()#dpi=4)
    #plt.savefig('./figs/corr_validation.png', dpi=400, bbox_inches='tight')

#Convert Coordinates
def convertCoords(df,tx,rx):
    earth_measurements=[6378,6356] #Earth Measurements
    #TODO: I need to verify these numbers as I currentley don't seem to be able to access the gitlab
    tx_xyz=[0,0,0]
    rx_xyz=[0,0,0]
    df['X'],df['Y'],df['Z']=jt.geodetic2aer.geodetic2ecef(df['lat'],df['lon'],.001*df['geoaltitude'],earth_measurements[0],earth_measurements[1])
    tx_xyz[0],tx_xyz[1],tx_xyz[2]=jt.geodetic2aer.geodetic2ecef(tx[0],tx[1],.001*tx[2],earth_measurements[0],earth_measurements[1])
    rx_xyz[0],rx_xyz[1],rx_xyz[2]=jt.geodetic2aer.geodetic2ecef(rx[0],rx[1],.001*rx[2],earth_measurements[0],earth_measurements[1])
    return df,tx_xyz,rx_xyz

#Find Bistatic range in Km
def bistaticrange(tx,rx,df):
    x=df['X']
    y=df['Y']
    z=df['Z']
    tx_range=np.sqrt((tx[0]-x)**2+(tx[1]-y)**2+(tx[2]-z)**2) #Calculate distance from plane to tx
    rx_range=np.sqrt((rx[0]-x)**2+(rx[1]-y)**2+(rx[2]-z)**2) #Calculate distance from plane to rx
    bistatic_range=tx_range+rx_range #Sum to return bistatic range
    df['brange']=bistatic_range

    #Plot bistatic range vs time
    plt.scatter(df['time'],df['brange'])
    plt.title("bistatic range vs time")
    plt.xlabel("Time (Unix Timestamp)")
    plt.ylabel("bistatic range (km)") 

    return df,bistatic_range #Return Bistatic range

#Function to cut data outside of 60 seconds of minimum bistatic range
def timeCut(df, time):
    bmin = min(df['brange'])
    min_index = df[df['brange'] == bmin].index
    time_clossest_approach = df.loc[min_index, 'time'].iloc[0]
    reduced_df = df[(df['time'] >= time_clossest_approach - (time/2)) & (df['time'] <= time_clossest_approach + (time/2))]

    plt.scatter(reduced_df['time'],reduced_df['brange'])
    plt.title("Reduced time bistatic range vs time")
    plt.xlabel("Time (Unix Timestamp)")
    plt.ylabel("bistatic range (km)")

    return reduced_df  # Optional: Return the filtered DataFrame

#Need Function that determines cycle number from bistatic range #4:39
def calcCyclePhase(df,wavelegnth,resolution,showgraph):
    df = df.copy().reset_index(drop=True)
    #Calculate cycles
    df["cycles"]=df['brange']/wavelegnth


    min_index=df.index.min()
    cycles_init=df['cycles'][min_index]

    df['phase']=cycles_init-df['cycles']

    #I realized I should save this until the interpolation is complete
    df['phase_wraping']=jt.interferometry.wrap_phase(2*3.14*df['phase'])

#TODO: Eventually I need to adjust this so that everything fits together in a subplot like above
    if (showgraph==True):
        plt.figure(1)
        plt.scatter(df['time'], df['cycles']) 
        plt.title("Cycle number vs Time")
        plt.xlabel("Time (Unix Timestamp)")
        plt.ylabel("# Phase Cycles")

        min_index=df.index.min()
        cycles_init=df['cycles'][min_index]

        plt.figure(2)
        df['phase']=cycles_init-df['cycles']
        plt.scatter(df['time'], df['phase']) 
        plt.title("Phase vs Time")
        plt.xlabel("Time (Unix Timestamp)")
        plt.ylabel("Phase")

        #I realized I should save this until the interpolation is complete
        df['phase_wraping']=jt.interferometry.wrap_phase(2*3.14*df['phase'])

        plt.figure(3)
        plt.scatter(df['time'], df['phase_wraping']) 
        plt.title("Phase Wrapping with wavelegnth 10 ft(I think)")
        plt.xlabel("Time (Unix Timestamp)")
        plt.ylabel("Phase Wrapping")

    timePhase_df=df[['time','phase']].copy()

    df_toInterpolate=pd.DataFrame(columns=['time','phase'])
    df_toInterpolate=df_toInterpolate.reset_index(drop=True)

    N=resolution*len(timePhase_df)
    x=0

    #Convert so that time index starts at 0
    #mintime=timePhase_df['time'].min()
    #timePhase_df['time']=timePhase_df['time']-mintime

    x=np.linspace(timePhase_df['time'][0], timePhase_df['time'][len(timePhase_df)-1], num=N)

    P=BarycentricInterpolator(timePhase_df['time'],timePhase_df['phase'])#Creates a function P which values can be put into

    phase=P(x)
    phase_wrap=jt.interferometry.wrap_phase(2*3.14*P(x))

    plt.figure(dpi=800)
    plt.scatter(x,phase_wrap,1)
    plt.title("Interpolated Wrapped Phase vs Time")
    plt.xlabel("Time (Unix Timestamp)")
    plt.ylabel("Phase Wrapping")

    interpolated_df=pd.DataFrame({'time_s':x,'phase':phase,'phaseWraping':phase_wrap}) #These values would be what you plug into the auto calibration

    return interpolated_df
#Function that interpolatesdatapoints to sufficient resolution

#Note that having all of this in one class could be usefull, but i need more time to think about how that would work