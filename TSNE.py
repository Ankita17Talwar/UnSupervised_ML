# Import TSNE
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

grain_data = pd.read_csv('dataset/Grains/seeds.csv')

#
samples = grain_data.iloc[:, :-1].values


variety_numbers = grain_data.iloc[:, -1:].values

print(variety_numbers)

# Create a TSNE instance: model
model = TSNE(learning_rate=200)

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(samples)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs,ys,c=variety_numbers)
plt.show()
