import pandas as pd
import numpy as np

tsb = pd.read_csv('log.csv')[-10:] # traffic state of blackbox after the model made predictions

print(tsb)

for i, row in tsb.iterrows():
    if ((row['north_queue_size']+row['south_queue_size'])-(row['east_queue_size']+row['south_queue_size']))>10:
        tsb.at[i,'north_light'] = 1
        tsb.at[i,'south_light'] = 1
        tsb.at[i,'east_light'] = 0
        tsb.at[i,'west_light'] = 0
    elif ((row['north_queue_size']+row['south_queue_size'])-(row['east_queue_size']+row['south_queue_size']))>10:
        tsb.at[i,'north_light'] = 0
        tsb.at[i,'south_light'] = 0
        tsb.at[i,'east_light'] = 1
        tsb.at[i,'west_light'] = 1
    elif i != min(tsb.index):
        tsb.at[i,'north_light'] = tsb.at[i-1,'north_light'].copy()
        tsb.at[i,'south_light'] = tsb.at[i-1,'south_light'].copy()
        tsb.at[i,'east_light'] = tsb.at[i-1,'east_light'].copy()
        tsb.at[i,'west_light'] = tsb.at[i-1,'west_light'].copy()

print(tsb)