#Script to read and plot ADSB data

import pandas as pd

#Path to CSV File providinig Flight's Database
filePath='/Users/katlingarula/Desktop/Current_CU_Stuff/DLA/Flight_Data/Test_Data.csv'


querry_string='callsign=="EZY764Z "'
#Ex     'callsign==FLIGHT" Replace Flight with calsign
            #All callsign variables will be 8 characters, the last 2 might be spaces
            #And will not show up in a spreadsheet


def fetch_flight_data(csv_path,querryString):
    df_all_flights = pd.read_csv(filePath,header='infer')
    print("Legnth of Dataset: ", len(df_all_flights))
    
    df_individual_flight=df_all_flights.query(querryString)
    print("Flight Specific Datapoints: ",len(df_individual_flight))
    
    return df_individual_flight

#Replace "Flight with the callsign of the plane of interest
#This will save said flight in .../DLA/FLIGHT.csv
FLIGHT_df=fetch_flight_data(filePath,querry_string)

#Uncomment and Change file name when ready to save flight data
#Note that the process of saving a file can take a few seconds to a minute
FLIGHT_df.to_csv('/Users/katlingarula/Desktop/Current_CU_Stuff/DLA/IndividualFlightData/EZY764Z.csv')

