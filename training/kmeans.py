from sklearn.cluster import KMeans
import h5py
import numpy as np

hf = h5py.File('../data/RPC.hdf5', 'r')

occupy_endcap_data = hf.get('ENDCAP')

kmeans_model = KMeans(n_clusters=2).fit(occupy_endcap_data)


