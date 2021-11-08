# cluster companies using their daily stock price movements
# (i.e. the dollar difference between the closing and opening prices for each trading day).
# You are given a NumPy array movements of daily price movements from 2010 to 2015 (obtained from Yahoo! Finance),
# where each row corresponds to a company, and each column corresponds to a trading day.
# DataSet used : company-stock-movements-2010-2015-incl.csv

from sklearn.preprocessing import Normalizer
#  Normalizer will separately transform each company's stock price to a relative scale before the clustering begins.
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd

# read file into numpy array
movements = pd.read_csv('dataset/company-stock-movements-2010-2015-incl.csv', header=0, index_col=0)
print(movements.head())

data = movements.values

companies = movements.index.values
print(companies)

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(data)

# Predict the cluster labels: labels
labels = pipeline.predict(data)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print(df.sort_values('labels'))