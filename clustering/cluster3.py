import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import h5py
import argparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--readfile')
parser.add_argument('--savefile')
parser.add_argument('--n_clusters')
parser.add_argument('--n_trajs')
args = parser.parse_args()
n_trajs = int(args.n_trajs)

hf = h5py.File(args.readfile, 'r')
# hfw = h5py.File(args.savefile + '_fftpca.h5', 'w')
# modelfile = '/projects/kumar-lab/mehtav/fft/' + args.savefile + '.sav'

key_list = [key for key in hf.keys()]
pointdata = []
fft_features = []
# train = []
for traj in range(n_trajs):
	key = key_list[traj]
	pointdata.append(hf.get(key+ '/points')[()])   # 240,7,2
	fft_features.append(hf.get(key+ '/fft_features')[()])
	# train = train + fft_features.tolist()

fft_features = np.array(fft_features)
train = np.reshape(fft_features, (fft_features.shape[0]*fft_features.shape[1], fft_features.shape[2]))
print(train.shape)

print('Fitting PCA')
explained_var = []
for i in range(1, 10):
	print(i)
	pca = PCA(n_components=i, random_state=0).fit(train)
	explained_var.append(pca.explained_variance_ratio_.sum())
print('Fitted')

plt.figure()
plt.plot(np.arange(1, 10), explained_var)
plt.xlabel('n_components')
plt.ylabel('explained_var')
plt.show()

# print('Fitting Kmeans')
# kmeans = KMeans(n_clusters=int(args.n_clusters), random_state=0).fit(train)
# print('Fitted')

# pickle.dump(kmeans, open(modelfile, 'wb'))

# labels = np.reshape(kmeans.labels_, (fft_features.shape[0], fft_features.shape[1]))
# print(labels.shape)

# for traj in range(n_trajs):
# 	key = key_list[traj]
# 	g = hfw.create_group(key)
# 	pointtraj = hf.get(key+ '/points')[()]   # 240,7,2
# 	fft_features = hf.get(key+ '/fft_features')[()]
# 	g.create_dataset('points', data = pointtraj)
# 	g.create_dataset('fft_features', data = fft_features)
# 	g.create_dataset('labels', data = labels[traj])

# hfw.close()