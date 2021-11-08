from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# read file into numpy array
movements = pd.read_csv('dataset/company-stock-movements-2010-2015-incl.csv', header=0, index_col=0)
print(movements.head())

data = movements.values

companies = movements.index.values
print(companies)

# Normalize the data (stock price movements)
normalized_data = normalize(data)

# Calculate the linkage: mergings
mergings = linkage(normalized_data, method='complete')

# Plot the dendrogram
dendrogram(mergings, labels=companies, leaf_rotation=90, leaf_font_size=6)
plt.show()



