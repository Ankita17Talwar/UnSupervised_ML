# Dataset -wine.csv
# Varities of Wine - Barolo, Grignolino and Barbera
# Featues measures chemical composition  eg alcohol content ; Visual properties - color intensity

from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

# load data from csv
wine_data = pd.read_csv('dataset/wine.csv')

# sample excluding columns classifying  wine
sample = wine_data.drop(['class_name','class_label'],axis=1).values

model = KMeans(n_clusters=3)
# print(model.fit_predict(sample))

wine_data['label'] = model.fit_predict(sample)

# cross_tabulation
print(pd.crosstab(wine_data['label'], wine_data['class_name']))

# class_name  Barbera  Barolo  Grignolino
# label
# 0                29      13          20
# 1                 0      46           1
# 2                19       0          50

# Standardize feature
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(sample)
StandardScaler(copy=True,with_mean=True, with_std=True)
sample_scaled = scaler.transform(sample)

# clusetr data using K-means

model = KMeans(n_clusters=3)
model.fit_predict(sample_scaled)
wine_data['label_new'] = model.fit_predict(sample_scaled)

# cross_tabulation
print(pd.crosstab(wine_data['label_new'], wine_data['class_name']))

# class_name  Barbera  Barolo  Grignolino
# label_new
# 0                48       0           3
# 1                 0      59           3
# 2                 0       0          65

# Above can also be achived using pipeline
