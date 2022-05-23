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
Dist = df.drop_duplicates(['Distance']);
Dist = Dist['Distance'].to_numpy();
Conc = df['NH4'].to_numpy();

# create figure with station and distance
fig, ax = plt.subplots(figsize=(8, 4));
ax.scatter(df['ID_Station'],df['Distance']);
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
cbar.ax.set_yticklabels(df.DateMonth);
cbar.set_label('Time');


# reshape data

Range_ID = np.arange(ID.min(),ID.max()+1,step=1);
data = df.pivot(index='DateMonth',columns='ID_Station',values='NH4');
Matrix = data.values
dataOne = data.stack()
index = dataOne.index
Date_num = np.linspace(1,len(data),len(data));

# Time serie plot
fig, ax2 = plt.subplots(figsize=(8, 4));
ax2.plot(Date_num,Matrix[:,9],label='Station 23');
ax2.plot(Date_num,Matrix[:,18],label='Station 32');
ax2.plot(Date_num,Matrix[:,1],label='Station 15');
ax2.set_xlabel('Number of days from 01-1993');
ax2.set_ylabel('Concentration');
ax2.set_title('Concentration of NH4 of few stations over time');
ax2.legend()



Xtime = np.array([Date_num,data.mean(axis=1).to_numpy()]);
Xtime = Xtime.transpose();
ytime = data.mean(axis=1).to_numpy();

Xspace = np.array([Dist,data.mean(axis=0).to_numpy()]);
Xspace = Xspace.transpose();
yspace = data.mean(axis=0).to_numpy();

# 
# split train and test sets
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42);

# import Kmeans
from sklearn.cluster import KMeans
#from sklearn.linear_model import LogisticRegression
 
rnd_seed = 2022
rnd_gen = np.random.default_rng(rnd_seed);

km1 = KMeans(random_state = rnd_seed); # Random seed
y_km_time = km1.fit_predict(Xtime); # Fitting to data subset

# # plot the 3 clusters with time
# fig, ax3 = plt.subplots(figsize=(16, 8));
# ax3.scatter(Xtime[y_km_time == 0, 0], Xtime[y_km_time == 0, 1],s=50, c='lightgreen',marker='s', edgecolor='black',label='regime 0')
# ax3.scatter(Xtime[y_km_time == 1, 0], Xtime[y_km_time == 1, 1],s=50, c='orange',marker='o', edgecolor='black',label='regime 1')
# ax3.scatter(Xtime[y_km_time == 2, 0], Xtime[y_km_time == 2, 1],s=50, c='lightblue',marker='v', edgecolor='black',label='regime 2')
# # plot the centroids
# ax3.scatter(km1.cluster_centers_[:, 0], km1.cluster_centers_[:, 1],s=250, marker='*',c='red', edgecolor='black',label='centroids')
# ax3.legend(scatterpoints=1)
# # apparence
# ax3.grid()
# ax3.set_xticks(np.arange(0, 324, step=12),
#                ['1993','1994','1995','1996','1997','1998','1999','2000','2001','2002',
#                 '2003','2004','2005','2006','2007','2008','2009','2010','2011','2012',
#                 '2013','2014','2015','2016','2017','2018','2019']);
# ax3.set_xlabel('Time [months]');
# ax3.set_ylabel('Concentration of NH4 [mg/l]');
# ax3.set_title('Regime on the entire river throught time');

# km2 = KMeans(n_clusters=3, # Number of clusters to split into 
#                       random_state = rnd_seed); # Random seed
# y_km_space = km2.fit_predict(Xspace); # Fitting to data subset
# # plot the 3 clusters with space
# fig, ax4 = plt.subplots(figsize=(16, 8));
# ax4.scatter(Xspace[y_km_space == 0, 0], Xspace[y_km_space == 0, 1],s=50, c='lightgreen',marker='s', edgecolor='black',label='regime 0')
# ax4.scatter(Xspace[y_km_space == 1, 0], Xspace[y_km_space == 1, 1],s=50, c='orange',marker='o', edgecolor='black',label='regime 1')
# ax4.scatter(Xspace[y_km_space == 2, 0], Xspace[y_km_space == 2, 1],s=50, c='lightblue',marker='v', edgecolor='black',label='regime 2')
# # plot the centroids
# ax4.scatter(km2.cluster_centers_[:, 0], km2.cluster_centers_[:, 1],s=250, marker='*',c='red', edgecolor='black',label='centroids')
# ax4.legend(scatterpoints=1)
# # apparence
# ax4.grid()
# # ax4.set_xticks(np.arange(Range, len(Range_ID), step=1),
# #                 [Range_ID.astype(str)]);
# ax4.set_xlabel('Distance from the source [km]');
# ax4.set_ylabel('Concentration of NH4 [mg/l]');
# ax4.set_title('Regime on the entire river throught space');

# # visualization
# model = LogisticRegression(solver = 'lbfgs', max_iter=10000)
# visualizer = ClassificationReport(model)

# visualizer.fit(X, y)
# #visualizer.score(X_test, y_test)
# #visualizer.show()


# from yellowbrick.cluster import KElbowVisualizer
# model = KMeans()

# # k is range of number of clusters.
# visualizer = KElbowVisualizer(model, k=(5,10),metric='silhouette', timings= True)
# visualizer.fit(X)        # Fit the data to the visualizer
# visualizer.show()        # Finalize and render the figure




