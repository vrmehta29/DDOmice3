#!/usr/bin/env python

from segmentcentroid.envs.MiceEnv import MiceEnv
from segmentcentroid.tfmodel.MiceNNModel import MiceNNModel
from segmentcentroid.planner.traj_utils import *

import numpy as np
import copy
import os
import pickle
import h5py

import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

def runPolicies(full_traj, 
		cov_mat,
		super_iterations=100,
		sub_iterations=0,
		learning_rate=1e-2,
		env_noise=0.3,
		file_name = '',
		n_start_states = 1):

	m  = MiceNNModel(10, statedim=(22, 1), actiondim = (22, 1), hidden_layer = 48, cov_mat = cov_mat)

	#full_traj = [] # List of trajs which are lists of (s,a) tuples

	m.sess.run(tf.initialize_all_variables())

	with tf.variable_scope("optimizer"):
		opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

		m.train(opt, full_traj, super_iterations, sub_iterations)

	saver = tf.train.Saver()
	saver.save(m.sess, '/projects/kumar-lab/mehtav/models/test')

	def generate_random_start_state():
		random_traj = full_traj[np.random.choice(len(full_traj), 1)]
		start_state = random_traj[np.random.choice(len(random_traj), 1)][0]
		print("Start state: ")
		print(start_state)
		return start_state
	# g = MiceEnv(start_state)
	
	hf = h5py.File('/projects/kumar-lab/mehtav/generated_traj/gt_tail.h5', 'w')
	# generated_traj = {}
	for i in range(m.k):
		g = hf.create_group('option'+str(i))
		for j in n_start_states:
			start_state = generate_random_start_state()
			print('Visualising option ' + str(i))
			traj = m.visualizePolicy(i, start_state, frames=600, trajectory_only = True)
			# generated_traj = traj
			g.create_dataset('traj'+str(j), data=np.array(traj))

	hf.close()

	# with open('/projects/kumar-lab/mehtav/generated_traj/gt_tail.pkl', 'wb') as f:
	# 	pickle.dump(generated_traj, f)


def main():
	
	CENTER_SPINE_INDEX = 6
	BASE_TAIL_INDEX = 9
	# Sampling 1000 traj randomly
	filelist = os.listdir('/projects/kumar-lab/mehtav/normalised_vd_tail_2')
	filelist = np.random.choice(filelist, 1000)

	data = []
	for file in filelist:
		with open('/projects/kumar-lab/mehtav/normalised_vd_tail_2/'+file, 'rb') as f:
			data.append(pickle.load(f))

	# data = [file_number, points/conf/vel, traj_number]
	# Each traj has shape (nrow x 12 x 2) or (nrow by 12)

	count = 0
	full_traj = []
	for file in range(len(data)):
		print(count)
		count = count+1
		for t in range(len(data[file][1])):
			p_traj = data[file][0][t]
			c_traj = data[file][1][t]
			v_traj = data[file][2][t]
			op_traj = []
			for i in range(v_traj.shape[0]):
				s = np.ravel(np.delete(p_traj[i], BASE_TAIL_INDEX, 0)) # Removing center spine, as it is always at the origin
				a = np.ravel(np.delete(v_traj[i], BASE_TAIL_INDEX, 0)) # Removing center spine, as it is always at rest	
				# s = np.delete(s, 17, 0) # Removing x coordinate of base tail
				# a = np.delete(a, 17, 0) # Removing x coordinate of base tail
				op_traj.append((s, a))
			full_traj.append(op_traj)


	runPolicies(full_traj, np.eye(22), n_start_states = 5) #np.load('/projects/kumar-lab/mehtav/cov.npy').astype(np.float32), file_name = 'LR1e0')


main()


