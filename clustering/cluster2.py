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
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA
import pickle
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--readfile', help = 'Of the format fftfeat.h5')
parser.add_argument('--savefile', help = 'Just the name, eg. 7')
parser.add_argument('--n_clusters', help = 'Number of clusters')
parser.add_argument('--n_trajs', help = 'Number of trajectories to consider for training clustering')

args = parser.parse_args()
n_trajs = int(args.n_trajs)

hf = h5py.File('/projects/kumar-lab/mehtav/fft/'+ args.readfile, 'r')
hfw = h5py.File('/projects/kumar-lab/mehtav/fft/'+ args.savefile + '_fftlabels.h5', 'w')
gmmsave = '/projects/kumar-lab/mehtav/fft/' + args.savefile + 'gmm' + '.sav'
pcasave = '/projects/kumar-lab/mehtav/fft/' + args.savefile + 'pca' + '.sav'

key_list = [key for key in hf.keys()]
pointdata = []
fft_features = []
time_features = []
traj_key_list = []
# train = []
for traj in range(n_trajs):
	print(traj)
	key = key_list[traj]
	# print(hf[key].keys())
	pointdata.append(hf.get(key+ '/points')[()])   # 240,7,2
	traj_key_list.append(hf.get(key+ '/traj_key')[()])
	fft_features.append(hf.get(key+ '/fft_features')[()])
	time_features.append(hf.get(key+ '/rangles')[()])

	# train = train + fft_features.tolist()

fft_features = np.array(fft_features)[:,:-1,:]
time_features = np.array(time_features)[:, 8:-8]
print(fft_features.shape, time_features.shape)
train = np.reshape(fft_features, (fft_features.shape[0]*fft_features.shape[1], fft_features.shape[2]*fft_features.shape[3]))
time_features = np.reshape(time_features, (time_features.shape[0]*time_features.shape[1], time_features.shape[2]))
print(train.shape)

print('Fitting PCA')
pca = PCA(n_components=6, random_state=0).fit(train)
train_pca = pca.transform(train)
print('Fitted with explained variance of ', pca.explained_variance_ratio_.sum())

pickle.dump(pca, open(pcasave, 'wb'))

print(train_pca.shape)

train_time_fft = np.concatenate((train_pca, time_features ), axis = 1)
print(train_time_fft.shape)

print('Fitting GMM')
gmm = GMM(n_components=int(args.n_clusters), random_state=0).fit(train_time_fft)
# kmeans = KMeans(n_clusters=int(args.n_clusters), random_state=0).fit(train_time_fft)
labels = gmm.predict(train_time_fft)
db_score = (sklearn.metrics.davies_bouldin_score(train_time_fft, labels))
print("Converged: ", str(gmm.converged_) )
print('Fitted with DB score: ', db_score)

pickle.dump(gmm, open(gmmsave, 'wb'))

labels = np.reshape(labels, (fft_features.shape[0], fft_features.shape[1]))
unique_elements, counts_elements = np.unique(labels, return_counts=True)
print(labels.shape)
print("Unique elements: ", unique_elements)
print("Frequency: ", counts_elements)

def list2string(list):
	string = ""
	for i in range(len(list)):
		string = string + str(list[i]) + ","
	return string

labels_string = [list2string(labels[i]) for i in range(labels.shape[0])]
video_labels = pd.DataFrame({"traj_key": traj_key_list, "label": labels_string})
video_labels.to_csv('/projects/kumar-lab/mehtav/fft/' + args.savefile + 'videolabels' + '.csv')

for traj in range(n_trajs):
	key = key_list[traj]
	g = hfw.create_group(key)
	pointtraj = hf.get(key+ '/points')[()]   # 240,7,2
	# fft_features = hf.get(key+ '/fft_features')[()]
	g.create_dataset('points', data = pointtraj)
	# g.create_dataset('fft_features', data = fft_features)
	g.create_dataset('labels', data = labels[traj])

hfw.close()