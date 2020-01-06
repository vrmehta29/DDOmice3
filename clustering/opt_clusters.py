import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import affine
import h5py
import argparse
import sklearn
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

hf = h5py.File('/projects/kumar-lab/mehtav/fft/'+ args.readfile, 'r')
hfw = h5py.File('/projects/kumar-lab/mehtav/fft/'+ args.savefile + '_fftlabels.h5', 'w')
kmeanssave = '/projects/kumar-lab/mehtav/fft/' + args.savefile + 'kmeans' + '.sav'
pcasave = '/projects/kumar-lab/mehtav/fft/' + args.savefile + 'pca' + '.sav'

key_list = [key for key in hf.keys()]
pointdata = []
fft_features = []
time_features = []
# train = []
for traj in range(n_trajs):
	key = key_list[traj]
	pointdata.append(hf.get(key+ '/points')[()])   # 240,7,2
	fft_features.append(hf.get(key+ '/fft_features')[()])
	time_features.append(hf.get(key+ '/angles')[()])
	# train = train + fft_features.tolist()

fft_features = np.array(fft_features)
time_features = np.array(time_features)[:, 8:-8]
print(fft_features.shape, time_features.shape)
train = np.reshape(fft_features, (fft_features.shape[0]*fft_features.shape[1], fft_features.shape[2]))
time_features = np.reshape(time_features, (time_features.shape[0]*time_features.shape[1], time_features.shape[2]))
print(train.shape)

print('Fitting PCA')
pca = PCA(n_components=6, random_state=0).fit(train)
train_pca = pca.transform(train)
print('Fitted')

pickle.dump(pca, open(pcasave, 'wb'))

print(train_pca.shape)

train_time_fft = np.concatenate((train_pca, time_features ), axis = 1)
print(train_time_fft.shape)

db_score = []

print('Fitting Kmeans')
for nc in range(5, 50):
	print(nc)
	kmeans = KMeans(n_clusters=nc, random_state=0).fit(train_time_fft)
	db_score.append(sklearn.metrics.davies_bouldin_score(train_time_fft, kmeans.labels_))
print('Fitted')

plt.figure()
plt.plot(np.arange(5,50), db_score)
plt.xlabel('n_clusters')
plt.ylabel('DB Score')
plt.savefig('/projects/kumar-lab/mehtav/fft/' + args.savefile + "db_score.png")

# print("DB Score for ", args.n_clusters, "clusters: ", sklearn.metrics.davies_bouldin_score(train_time_fft, kmeans.labels_))

pickle.dump(kmeans, open(kmeanssave, 'wb'))

labels = np.reshape(kmeans.labels_, (fft_features.shape[0], fft_features.shape[1]))
print(labels.shape)

for traj in range(n_trajs):
	key = key_list[traj]
	g = hfw.create_group(key)
	pointtraj = hf.get(key+ '/points')[()]   # 240,7,2
	fft_features = hf.get(key+ '/fft_features')[()]
	g.create_dataset('points', data = pointtraj)
	g.create_dataset('fft_features', data = fft_features)
	g.create_dataset('labels', data = labels[traj])

hfw.close()