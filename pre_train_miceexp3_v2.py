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
import tensorflow as tf
import warnings
from segmentcentroid.envs.MiceEnv import MiceEnv
from segmentcentroid.tfmodel.MiceNNModel import MiceNNModel
from segmentcentroid.planner.traj_utils import *

warnings.filterwarnings("module")

parser = argparse.ArgumentParser()
parser.add_argument('--readfile', help = 'fftfeat.h5')	#Read fftfeat file
parser.add_argument('--savefile')
parser.add_argument('--kmeansmodel')
parser.add_argument('--pca')
parser.add_argument('--hidden_layer')	#512,1024,512
parser.add_argument('--epochs')
parser.add_argument('--ntrajs')
parser.add_argument('--options')
parser.add_argument('--model')
parser.add_argument('--lr')
parser.add_argument('--init_weights')
parser.add_argument('--tb')
args = parser.parse_args()
ntrajs = int(args.ntrajs)

hf = h5py.File(args.readfile, 'r')
pca = pickle.load(open(args.pca, 'rb'))
kmeans = pickle.load(open(args.kmeansmodel, 'rb'))

n_clusters = kmeans.means_.shape[0]

key_list = [key for key in hf.keys()]
pointdata = []
fft_features = []
time_features = []
veldata = []
rangles = []
data = []
mu = 0.0	#Average relative velocity
mu_count = 0.0
for traj in range(ntrajs):
	print(traj)
	key = key_list[traj]
	# print([k for k in hf['LL6-B2B+2018-06-04_SPD+C57BR_Male_154-155-3-PSY_pose_est_v2_789'] ])
	# print(hf.get(key+ '/points')
	pointtraj = (hf.get(key+ '/points')[()])   # 240,7,2
	fft_features.append(hf.get(key+ '/fft_features')[()])
	rangles.append(hf.get(key+ '/rangles')[()])
	time_features.append(hf.get(key+ '/rangles')[()]) #Will be modified later, hence making 2 copies of angular data
	velocitytraj = (hf.get(key+ '/velocity')[()])
	pointdata = pointdata[8:-8]

	ang_velocity = []
	for frame in range(rangles[traj].shape[0]-1):
		ang_velocity.append([rangles[traj][frame+1][angle] - rangles[traj][frame][angle] for angle in range(len(rangles[traj][frame]))])
		mu = mu_count/(mu_count+1)*mu + np.mean(ang_velocity)/(mu_count+1)
		mu_count = mu_count+1
	ang_velocity = np.array(ang_velocity)

	traj = [pointtraj, velocitytraj, ang_velocity]
	data.append(traj)
	# train = train + fft_features.tolist()

print("Average relative angular velocity: ", mu, mu_count, flush = True)

fft_features = np.array(fft_features)
fft_features = np.reshape(fft_features, (fft_features.shape[0], fft_features.shape[1], fft_features.shape[2]*fft_features.shape[3]))
print(np.array(time_features).shape)
time_features = np.array(time_features)[:,8:-7,:]
print(fft_features.shape, time_features.shape)
train = np.reshape(fft_features, (fft_features.shape[0]*fft_features.shape[1], fft_features.shape[2]))
time_features = np.reshape(time_features, (time_features.shape[0]*time_features.shape[1], time_features.shape[2]))
print(train.shape, time_features.shape)

print('Fitting PCA')
# pca = PCA(n_components=6, random_state=0).fit(train)
train_pca = pca.transform(train)
print('Fitted')
print(train_pca.shape, time_features.shape)

train_time_fft = np.concatenate((train_pca, time_features ), axis = 1)
print(train_time_fft.shape)

print('Fitting Kmeans')
# kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(train_time_fft)
db_score = (sklearn.metrics.davies_bouldin_score(train_time_fft, kmeans.predict(train_time_fft)))
print('Fitted with DB score: ', db_score)

labels = np.reshape(kmeans.predict(train_time_fft), (fft_features.shape[0], fft_features.shape[1]))
train_time_fft = np.reshape(train_time_fft, (fft_features.shape[0], fft_features.shape[1], -1))
unique_elements, counts_elements = np.unique(labels, return_counts=True)
print('state shape', train_time_fft.shape)
print(labels.shape)
print(unique_elements)
print(counts_elements)

# time_features = np.reshape(time_features, (time_features.shape[0], time_features.shape[1], time_features.shape[2]))






N_FRAMES = 16
N_FRAMES_AFTER = 4

def runPolicies(full_traj, 
		cov_mat,
		super_iterations=int(args.epochs)*int(args.ntrajs),
		sub_iterations=0,
		learning_rate= 10**(float(args.lr)),
		env_noise=0.3,
		file_name = '',
		n_start_states = 1):

	hidden_layer = [int(i) for i in args.hidden_layer.split(',')]
	m  = MiceNNModel(int(args.options), statedim=(12*N_FRAMES, 1), actiondim = (6*N_FRAMES_AFTER, 1)
					, hidden_layer = hidden_layer, cov_mat = cov_mat, init_weights = args.init_weights, learning_rate = learning_rate)

	#full_traj = [] # List of trajs which are lists of (s,a) tuples

	m.sess.run(tf.global_variables_initializer())

	with tf.variable_scope("optimizer"):
		opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

		m.train(opt, full_traj, int(args.ntrajs), super_iterations, sub_iterations, 25, args.tb)
	
	saver = tf.train.Saver()
	saver.save(m.sess, os.path.join(args.tb, args.model, args.model))



count = 0
full_traj = []
for file in range(len(data)):
	print(count)
	count = count+1
	# print(data[file][0][0].shape)
	p_traj = data[file][0][8:-8]
	v_traj = data[file][1][8:-7]
	w_traj = data[file][2][8:-7]
	labels_traj = labels[file]
	train_time_fft_traj = train_time_fft[file]
	op_traj = []
	# print(p_traj.shape, v_traj.shape, labels_traj.shape)
	for i in range(N_FRAMES-1, v_traj.shape[0] - N_FRAMES_AFTER):
		# s_ = p_traj[i- N_FRAMES+1 : i+1]
		s_ = train_time_fft_traj[i- N_FRAMES+1 : i+1]
		s = np.ravel(s_) # Taking the last 16 frames as state
		# a = np.ravel(v_traj[i: i+ N_FRAMES_AFTER]) # Removing center spine, as it is always at rest	
		a = np.ravel(w_traj[i: i+ N_FRAMES_AFTER])
		l = labels_traj[i]
		# s = np.delete(s, 17, 0) # Removing x coordinate of base tail
		# a = np.delete(a, 17, 0) # Removing x coordinate of base tail
		# print(a.shape)
		op_traj.append((s, a, l))
	full_traj.append(op_traj)


runPolicies(full_traj, 5*np.eye(6* N_FRAMES_AFTER), n_start_states = 5)