# -*- coding: utf-8 -*-
"""
Project Machine Learning - 2022

Vincenzo Guzzardi
"""
# import library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

# import data csv
df = pd.read_csv (r'C:\Users\vince\Documents\Unil\2. Master\1e_semestre2\Machine learning\Project\Data_ammonium\PB_1996_2019_NH4.csv',delimiter=(';'))
#train = pd.read_csv (r'C:\Users\vince\Documents\Unil\2. Master\1e_semestre2\Machine learning\Project\Data_ammonium\train.csv',delimiter=(','))
#test = pd.read_csv (r'C:\Users\vince\Documents\Unil\2. Master\1e_semestre2\Machine learning\Project\Data_ammonium\test.csv',delimiter=(','))

ID   = df.ID_Station.to_numpy();
Dist = df.Distance.to_numpy();
Conc = df.NH4.to_numpy();
df['DateTime'] = pd.to_datetime(df['Date'],infer_datetime_format=True);    
Date = (df['DateTime'] - df['DateTime'].min())  / np.timedelta64(1,'D');
Date = Date.to_numpy();

# function to convert day to datetime
def to_date(day):
    Date = day * np.timedelta64(1,'D') + df['Date'].min()
    return Date

# create figure with station and distance
fig, ax = plt.subplots(figsize=(8, 4));
ax.scatter(ID,Dist);
ax.set_xlabel('Station ID');
ax.set_ylabel('Distance form source [km]');
ax.set_title('Location of stations of measure');

# create figure with station and concentration
fig, ax = plt.subplots(figsize=(8, 4));
s = ax.scatter(ID,Conc,c = df.DateTime);
ax.set_xlabel('Station ID');
ax.set_ylabel('Concentration of NH4 [mg/l]');
ax.set_title('Concentrations of NH4 for each station');
cbar = plt.colorbar(s);
# cbar.ax.set_ticks([Date.min(),Date.max()])
# cbar.ax.set_yticklabels(df.Date);
cbar.set_label('Time');

# reshape data


# split train and test sets
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(ID,Conc,test_size=0.2,random_state=42);

# # import Kmeans
# from sklearn.cluster import KMeans

# rnd_seed = 2022
# rnd_gen = np.random.default_rng(rnd_seed);

# kmeans_test = KMeans(n_clusters=3, # Number of clusters to split into 
#                       random_state = rnd_seed); # Random seed
# kmeans_test.fit(X_train); # Fitting to data subset


