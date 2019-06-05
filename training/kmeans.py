from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import h5py
import numpy as np
import matplotlib.pyplot as plt

COLORS = ['green', 'blue']
LABELS = ['A', 'B']

hf = h5py.File('../data/RPC.hdf5', 'r')

occupy_endcap_data = hf.get('ENDCAP')

# KMeans
kmeans_model = KMeans(n_clusters=2).fit(occupy_endcap_data)
y_pred = kmeans_model.predict(occupy_endcap_data)

# PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(occupy_endcap_data)

# visualzie
plt.figure()

for index, pc in enumerate(principal_components):
    plt.scatter(
        pc[0], pc[1], alpha=0.8,
        color = COLORS[y_pred[index]],
        label = LABELS[y_pred[index]]
    )

plt.title('Clustering by K-Means in principal basis (RPC/ENDCAP sub-detector)')
plt.show()