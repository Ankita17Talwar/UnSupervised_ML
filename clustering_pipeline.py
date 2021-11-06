from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import pandas as pd


wine_data = pd.read_csv('dataset/wine.csv')

# sample excluding columns classifying  wine
sample = wine_data.drop(['class_name','class_label'],axis=1).values

model = KMeans(n_clusters=3)
scaler = StandardScaler()

pipeline = make_pipeline(scaler, model)
pipeline.fit(sample)
labels = pipeline.predict(sample)

wine_data['label'] = labels

# cross_tabulation
print(pd.crosstab(wine_data['label'], wine_data['class_name']))

# class_name  Barbera  Barolo  Grignolino
# label
# 0                 0       0          65
# 1                48       0           3
# 2                 0      59           3
