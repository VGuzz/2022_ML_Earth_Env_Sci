# -*- coding: utf-8 -*-
"""
Project Machine Learning - 2022

Prediction of Ammonium concentration in a river using
unsupervised Machine Learning

Vincenzo Guzzardi, 2022

Some parts come from the Notebook 4.2 of the ML course
by T. Beucler and M. Gomez, 2022
https://github.com/VGuzz/2022_ML_Earth_Env_Sci/blob/main/S4_2_Clustering_exercises_Vincenzo.ipynb

"""
#%% ======================= Importations =======================
# import library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# import data csv
df = pd.read_csv (r'C:\Users\vince\Documents\Unil\2. Master\1e_semestre2\Machine learning\Project\Data_ammonium\PB_1996_2019_NH4.csv',
                  delimiter=(';'))

#%% ======================= Preprocessing =======================

# Transform Date into Months
df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True); # Convert string date into Datetime
df['DateMonth'] = df['Date'].dt.to_period('M') # Convert into monthly time series

df = df.drop_duplicates(['ID_Station', 'DateMonth']) # remove duplicates

# Separate columns into variables
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

# reshape data
data = df.pivot(index='DateMonth',columns='ID_Station',values='NH4');
MatrixNan = data.values
dataOneCol = data.stack()

# Remove nan values
Median_station = np.nanmedian(MatrixNan,axis=0)

median_dict={}
for idx, Station_ID in np.ndenumerate(np.arange(14,35+1,1)):
    median_dict[Station_ID] = Median_station[idx]

Filled_data = data.fillna(value=median_dict)
Matrix = Filled_data.values

Matrix_tran = Matrix.transpose()

# create array with ID and numeric date
Range_ID = np.arange(ID.min(),ID.max()+1,step=1);
Date_num = np.linspace(1,len(data),len(data));

# Time serie plot
fig, ax2 = plt.subplots(2,1,figsize=(14, 8));
ax2[0].scatter(Date_num,MatrixNan[:,9],label='Station 23');
ax2[0].scatter(Date_num,MatrixNan[:,18],label='Station 32');
ax2[0].scatter(Date_num,MatrixNan[:,1],label='Station 15');
ax2[0].set_ylabel('Concentration [mg/l]');
ax2[0].set_title('With NaN values');
ax2[0].legend()
ax2[0].set_xticks(np.arange(0, 324, step=12),
                ['1993','1994','1995','1996','1997','1998','1999','2000','2001','2002',
                '2003','2004','2005','2006','2007','2008','2009','2010','2011','2012',
                '2013','2014','2015','2016','2017','2018','2019']);

ax2[1].scatter(Date_num,Matrix[:,9],label='Station 23');
ax2[1].scatter(Date_num,Matrix[:,18],label='Station 32');
ax2[1].scatter(Date_num,Matrix[:,1],label='Station 15');
ax2[1].set_xlabel('Time [Months]');
ax2[1].set_ylabel('Concentration [mg/l]');
ax2[1].set_title('Replaced by median');
ax2[1].legend()
ax2[1].set_xticks(np.arange(0, 324, step=12),
                ['1993','1994','1995','1996','1997','1998','1999','2000','2001','2002',
                '2003','2004','2005','2006','2007','2008','2009','2010','2011','2012',
                '2013','2014','2015','2016','2017','2018','2019']);
plt.subplots_adjust(hspace=0.25)

#%% ======================= Kmeans clustering =======================


k = 3 # number of clusters to test
rnd_seed = 2
rnd_gen = np.random.default_rng(rnd_seed);

km1 = KMeans(n_clusters=k,random_state = rnd_seed); # Random seed
y_km1 = km1.fit_predict(Matrix); # Fitting to data

km2 = KMeans(n_clusters=k,random_state = rnd_seed); # Random seed
y_km2 = km2.fit_predict(Matrix_tran); # Fitting to data

# Training model with k from 2 to 6

k_list = list(range(2,7,1))

km_models1 = []
km_models2 = []

for k in k_list:

    kmeans1 = KMeans(n_clusters=k, # Set the number of clusters 
                      random_state = rnd_seed) # Set the random state
    kmeans1.fit(Matrix) # Fit the model to data
    km_models1.append(kmeans1) # store the model trained to predict  k clusters
    
    kmeans2 = KMeans(n_clusters=k, # Set the number of clusters 
                      random_state = rnd_seed) # Set the random state
    kmeans2.fit(Matrix) # Fit the model to data
    km_models2.append(kmeans2) # store the model trained to predict  k clusters

#%% ======================= Metrics =======================

# Over time 

# Silhouette scores metric

silhouette_scores = [silhouette_score(Matrix, model.labels_)
                     for model in km_models1[:]]

best_index = np.argmax(silhouette_scores)
best_k = k_list[best_index]
best_score = silhouette_scores[best_index]

# Inertia metric
inertias = [model.inertia_ for model in km_models1[:]]
best_inertia = inertias[best_index]

# Figure of the metrics

fig, ax = plt.subplots(2,1,figsize=(8,4))
ax[0].plot(k_list, silhouette_scores, "bo-") 
ax[0].set_ylabel("Silhouette", fontsize=14)
ax[0].plot(best_k, best_score, "rs")
ax[0].set_title('Silhouette score',fontsize=16)

ax[1].plot(k_list, inertias, "bo-") 
ax[1].plot(best_k, best_inertia, "rs")
ax[1].set_xlabel("$k$", fontsize=14) 
ax[1].set_ylabel("Inertia", fontsize=14)
ax[1].set_title('Inertia score',fontsize=16)
plt.subplots_adjust(hspace=0.6)

# Over space ===> not working <===

# Silhouette scores metric

# silhouette_scores = [silhouette_score(Matrix_tran, model.labels_)
#                      for model in km_models2[:]]

# best_index = np.argmax(silhouette_scores)
# best_k = k_list[best_index]
# best_score = silhouette_scores[best_index]

# fig, ax = plt.subplots(figsize=(18,6))
# ax.plot(k_list, silhouette_scores, "bo-") 
# ax.set_xlabel("$k$", fontsize=14) 
# ax.set_ylabel("Silhouette score", fontsize=14)
# ax.plot(best_k, best_score, "rs")

# # Inertia metric
# inertias = [model.inertia_ for model in km_models2[:]]
# best_inertia = inertias[best_index]

# fig, ax = plt.subplots(figsize=(18,6))
# ax.plot(k_list, inertias, "bo-") 
# ax.plot(best_k, best_inertia, "rs")
# ax.set_xlabel("$k$", fontsize=14) 
# ax.set_ylabel("Inertia", fontsize=14)

#%% ======================= Visualization of clusters =======================

fig, ax = plt.subplots(figsize=(16, 6));
ax.scatter(Date_num, y_km1 );
ax.set_xlabel('Time [Months]',fontsize=14);
ax.set_ylabel('Regime of concentration',fontsize=14);
ax.set_title('Clustering of the concentration over the time',fontsize=16);
ax.set_xticks(np.arange(0, 324, step=12),
                ['1993','1994','1995','1996','1997','1998','1999','2000','2001','2002',
                '2003','2004','2005','2006','2007','2008','2009','2010','2011','2012',
                '2013','2014','2015','2016','2017','2018','2019'],fontsize=11);
ax.set_yticks(np.arange(0, 3, step=1),fontsize=11);

fig, ax = plt.subplots(figsize=(10, 4));
ax.scatter(Range_ID, y_km2 );
ax.set_xlabel('Station ID');
ax.set_ylabel('Regime of concentration');
ax.set_title('Clustering of the concentration over the space');
ax.set_yticks(np.arange(0, 3, step=1));



