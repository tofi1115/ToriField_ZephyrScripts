#Imports
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt # Plotting 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

#TODO: Add the video this came from here
def cfar_fast(
    x: np.ndarray,
    num_ref_cells: int,
    num_guard_cells: int,
    bias: float = 1,
    method=np.mean,
):
    '''
    CFAR function used for evaluating single time stamp in detect_plane
    Based off of https://www.youtube.com/watch?v=BEg29UuZk6c
    ------------
    Inputs:
        x: numpy ndarray,
            1D array to be searched for peaks
        num_ref_cells: int, 
            The number of reference/buffer cells
        num_guard_cells: int,
            The number of cells used in averaging to determine threshold
        bias: float
            bias to shift threshold up, (Should be greater than 1)
        method: function, 
            averaging method used, np.mean, np.median, np.min, np.max
    Outputs:
        window_mean: numpy array,
            1D array containing set of values, above which is considered a target 
    '''
    pad = int((num_ref_cells + num_guard_cells))
    # fmt: off
    window_mean = np.pad(                                                                   # Pad front/back since n_windows < n_points
        method(                                                                             # Apply input method to remaining compute cells
            np.delete(                                                                      # Remove guard cells, CUT from computation
                sliding_window_view(x, (num_ref_cells * 2) + (num_guard_cells * 2)),        # Windows of x including CUT, guard cells, and compute cells
                np.arange(int(num_ref_cells), num_ref_cells + (num_guard_cells * 2) + 1),   # Get indices of guard cells, CUT
                axis=1),
            axis=1
        ), (pad - 1, pad),
        constant_values=(np.nan, np.nan)                                                    # Fill with NaNs
    ) * bias                                                                                # Multiply output by bias over which cell is not noise
    # fmt: on

    return window_mean



def detect_plane(rti,
                start_samples:0,
                detect_options:{},
                optional_plots:[]):
    '''
    Function which detects and returns list of planes detected
    -------------
    Inputs:
        rti: 
        start_samples: int,
            Initial sample number, used in finding accurate values of sample_number, and 1d_idx
        detect_options: Dict,
            optional detection parameters, when left blank, function will use default tuning
            num_ref_cells: int, 
                number of reference cells
            num_guard_cells: int, 
                number of guard cells
            bias: float, 
                multiplicitive bias to threshold values
            gate_range: int tuple, 
                (min gate, max gate) describing range of rti gates to focus on
            edge_method: string, 
                method used to deal with edges in CFAR algorithm. Options: zero_padding, mean_padding, median_padding, 
                symetric, wrap, mean, median
            method: numpy function,
                cell averaging method used in CFAR algorithm ex: np.mean, np.min. np.max, np.median
            eps: float,
                Radius used in DBSCAN clustering
            min_samples: int,
                minimum number of points in neighborhood to be considered a cluster
            y_scaling: float,
                Scaling factor along y axis, can be used to preevent planes at close altitudes 
                from being identified as a single object
        optional_plots: list,
            [time of interest, 'CFAR_plot','plot_detections','DBSCAN']

    Outputs:
        list_detections: list,
            list of tuples: (gate, time, 1D_idx, sample_number)
        img: returns 0
        list_dict_properties: list,
            list containing dictionaries for each plane detection, with basic information on it
        list_peak_idx_1D: list,
            list of idx_1D for each detection event
    ----------------------
    Function works by using CFAR algorithm to find potential detections along timestamps, and then uses the 
    DBSCAN clustering algorithm to cluster detection events into planes, and eliminate outliers.
    '''
    
    #Set default detection options (pulled from detect_meteor function)
    detect_options_default = {"num_ref_cells":10,"num_guard_cells":3,"bias":1.9,"gate_range":(0,200),"edge_method":'symetric',
                            "method":np.median,"eps":55,"min_samples":5,"y_scaling":40}
    detect_options = {**detect_options_default, **detect_options}

    rti_shape=rti.shape                                                             #record shape of rti data
    list_detections=[]                                                              #initialize list_detections to empty list

    sample_time=0
    while sample_time<(rti_shape[1]):                                               # Loop though each time stamp in rti data
        padding_size=detect_options["num_guard_cells"]+detect_options["num_ref_cells"]  #Determine width of data with padding

    # Options for padding type of rti data
        if detect_options["edge_method"]=="zero_padding": #Pad data with 0's
            padded_RTI=rti[:,sample_time].pad(gate=padding_size,constant_values=0)

        if detect_options["edge_method"]=="mean_padding": #Pad rti with mean of rti at time stamp
            padded_RTI=rti[:,sample_time].pad(gate=padding_size,mode="mean")

        if detect_options["edge_method"]=="median_padding": #Pad rti with median of rti at time stamp
            padded_RTI=rti[:,sample_time].pad(gate=padding_size,mode="median")

        if detect_options["edge_method"]=="symetric": #Pad rti with symetric reflection of self at time stamp
            padded_RTI=rti[:,sample_time].pad(gate=padding_size, mode="symmetric")

        if detect_options["edge_method"]=="wrap": #Pad rti with edges wraped around
            padded_RTI=rti[:,sample_time].pad(gate=padding_size,mode="wrap")

        #use CFAR algorithm to find threshold for a single timestamp
        threshold=cfar_fast(padded_RTI,
                            num_guard_cells=detect_options["num_guard_cells"], 
                            num_ref_cells=detect_options["num_ref_cells"], 
                            bias=detect_options["bias"],
                            method=detect_options["method"])
        
        threshold=threshold[padding_size:-padding_size] #remove padding from threshold

        #Additional Code to account for edge cases
        if detect_options["edge_method"]== "mean": # Threshold on edges set equal to it's own mean
            threshold[np.isnan(threshold)] = np.nanmean(threshold)

        if detect_options["edge_method"]=="median": #Threshold on edges set equal to it's own median
            threshold[np.isnan(threshold)] = np.nanmedian(threshold)

        #Create targets_only array only including target rti values
        targets_only = np.copy(rti[:,sample_time])
        targets_only[np.where(rti[:,sample_time] < threshold)] = np.ma.masked

        #Optional Plot of CFAR on single time stamp
        if 'CFAR_plot' in optional_plots and sample_time==optional_plots[0]:
            plt.figure()
            plt.plot(rti[:,sample_time],c='b') #Plots RTI data
            plt.plot(threshold,label="Threshold",c='r') #Plots Threshold in Red
            plt.plot(targets_only,c='g') #Plots target values in green
            plt.xlim(detect_options["gate_range"])
            plt.show
        sample_time+=1

        gates=np.nonzero(targets_only !=0) #Create list of gates with visible targets
        sample_gate=0

        # Create list_detections with formating (gate,time,1D_idx, sample_number)
        while sample_gate<len(gates[0]):
            if detect_options["gate_range"][0]<=gates[0][sample_gate]<=detect_options["gate_range"][1]:
                tuple=(gates[0][sample_gate],sample_time,sample_gate*rti_shape[1]+sample_time+start_samples,rti_shape[0]*sample_time+sample_gate+start_samples) #Gate number, time, 1D_idx, sample_number
                list_detections.append(tuple)
            sample_gate+=1

    if len(list_detections)==0: #Check if possible ditections found

        print("No Detections Found") #Print Error message
        list_peaks_idx_1D=[]

        return(list_detections,0,list_dict_properties,list_peaks_idx_1D) #Return empty lists
    
    x_coords, y_coords ,idx_cooords,list_peaks_idx_1D= zip(*list_detections) #Break list_detections tuple into individual pieces for ploting
    if'plot_detections' in optional_plots: #optional plot detections graph
        plt.figure()
        plt.scatter(y_coords,x_coords,s=1) #Still need to figure out what to do about the edge cases, they show up prominently here
        plt.xlabel("time")
        plt.ylabel("gate")
        plt.title("Detections across RTI data")
        
#Clustering Algorithm
#Based on code from : https://www.geeksforgeeks.org/dbscan-clustering-in-ml-density-based-clustering/
    x_coords=np.multiply(x_coords,detect_options["y_scaling"])              #Scale Y axis to prevent Grouping between multiple gates
    DBSCAN_array=np.array(list(zip(y_coords,x_coords)))                     #Prepare Coords for use in DBSCAN
    db = DBSCAN(eps=detect_options["eps"], min_samples=detect_options["min_samples"]).fit(DBSCAN_array) #run DBSCAN
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    unique_labels = set(labels)

    #Optional plot of colored clusterings from DBSCAN
    if "DBSCAN" in optional_plots:
        colors = ['y', 'b', 'g', 'r']
        plt.figure()
        plt.scatter(y_coords,x_coords,c='black',s=1)
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = 'k'
            class_member_mask = (labels == k)
            xy = DBSCAN_array[class_member_mask & core_samples_mask]
            plt.scatter(xy[:, 0], xy[:, 1], c=col,s=1)
            xy = DBSCAN_array[class_member_mask & ~core_samples_mask]
            plt.scatter(xy[:, 0], xy[:, 1], c=col,s=1)
        plt.title('number of clusters: %d' % n_clusters_)
        plt.show()

    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    unique_labels = list(unique_labels)

    # Create a dictionary to store points for each cluster
    clustered_points = {}

    # Iterate through unique labels and get points for each cluster
    for label in unique_labels:
        clustered_points[label] = DBSCAN_array[labels == label]

    #Fill up dictionary with information on each cluster
    cluster=0
    list_dict_properties=[]

    while cluster< n_clusters_:
        plane=clustered_points[cluster]
        max_time=np.max(plane[:,0])
        min_time=np.min(plane[:,0])
        max_gate=np.max(plane[:,1])/detect_options["y_scaling"]
        min_gate=np.min(plane[:,1])/detect_options["y_scaling"]

        plane_info={"plane_number":cluster,"max_time":max_time,"min_time":min_time,"max_gate":max_gate,"min_gate":min_gate}
        list_dict_properties.append(plane_info)
        cluster+=1

    #Set Unused returns to 0
    img=0
    return(list_detections,img,list_dict_properties,list_peaks_idx_1D)


