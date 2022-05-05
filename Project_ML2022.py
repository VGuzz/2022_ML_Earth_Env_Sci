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

df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True); # Convert string date into Datetime
df['DateMonth'] = df['Date'].dt.to_period('M') # Convert into monthly time series
df = df.drop_duplicates(['ID_Station', 'DateMonth'])
ID   = df['ID_Station'].to_numpy();
Dist = df['Distance'].to_numpy();
Conc = df['NH4'].to_numpy();


# create figure with station and distance
fig, ax = plt.subplots(figsize=(8, 4));
ax.scatter(ID,Dist);
ax.set_xlabel('Station ID');
ax.set_ylabel('Distance form source [km]');
ax.set_title('Location of stations of measure');

# create figure with station and concentration
fig, ax = plt.subplots(figsize=(8, 4));
s = ax.scatter(ID,Conc,c = df['Date']);
ax.set_xlabel('Station ID');
ax.set_ylabel('Concentration of NH4 [mg/l]');
ax.set_title('Concentrations of NH4 for each station');
cbar = plt.colorbar(s);
# cbar.ax.set_ticks([Date.min(),Date.max()])
# cbar.ax.set_yticklabels(df.DateMonth);
cbar.set_label('Time');

# reshape data
# data = pd.DataFrame(index=pd.Series(pd.period_range("1/1/1993", freq="M", periods=322)),
#                     columns=range(ID.min(),ID.max()+1))
# Range_date = pd.period_range("1/1/1993", freq="M", periods=322);
# Range_date = np.arange('1993-01', '2019-11', dtype='datetime64[M]')
# Range_date=pd.period_range(start ='1993-01',
#               end ='2019-11', freq ='M')
# Range_ID = range(ID.min(),ID.max()+1);
# data = np.zeros((len(Range_date),len(Range_ID)));

# for i in range(0,13):
#     for j in range(0,321):
#         index = df.index[(df['ID_Station']==Range_ID[i]) & 
#                           (df['DateMonth']==Range_date[j])]# find the row number where ID=i & DateMonth = Range_date[j]
#         if index >= 0:
#             data[j,i] = Conc[index]
#         else:
#             data[j,i] = 0

data = df.pivot(index='DateMonth',columns='ID_Station',values='NH4');

X = data.index.astype(str)
y = data.mean(axis=1).to_numpy()
# 
# split train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42);

# import Kmeans
from sklearn.cluster import KMeans

rnd_seed = 2022
rnd_gen = np.random.default_rng(rnd_seed);

kmeans_test = KMeans(n_clusters=3, # Number of clusters to split into 
                      random_state = rnd_seed); # Random seed
kmeans_test.fit(X_train); # Fitting to data subset


