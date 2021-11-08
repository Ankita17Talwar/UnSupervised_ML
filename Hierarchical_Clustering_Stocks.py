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

## TSNE

# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=50)

# Apply fit_transform to normalized_data: tsne_features
tsne_features = model.fit_transform(normalized_data)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1th feature: ys
ys = tsne_features[:,1]

# Scatter plot
plt.scatter(xs,ys,alpha=0.5)


# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()


