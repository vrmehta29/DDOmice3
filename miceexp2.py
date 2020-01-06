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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_layer')
parser.add_argument('--epochs')
parser.add_argument('--ntrajs')
parser.add_argument('--options')
parser.add_argument('--model')
parser.add_argument('--lr')
parser.add_argument('--init_weights')
parser.add_argument('--tb')
args = parser.parse_args()

N_FRAMES = 16

def runPolicies(full_traj, 
		cov_mat,
		super_iterations=int(args.epochs)*int(args.ntrajs),
		sub_iterations=0,
		learning_rate= 10**(float(args.lr)),
		env_noise=0.3,
		file_name = '',
		n_start_states = 1):

	hidden_layer = [int(i) for i in args.hidden_layer.split(',')]
	m  = MiceNNModel(int(args.options), statedim=(14*N_FRAMES, 1), actiondim = (14, 1)
					, hidden_layer = hidden_layer, cov_mat = cov_mat, init_weights = args.init_weights)

	#full_traj = [] # List of trajs which are lists of (s,a) tuples

	m.sess.run(tf.global_variables_initializer())

	with tf.variable_scope("optimizer"):
		opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

		m.train(opt, full_traj, super_iterations, sub_iterations, 25, args.tb)
	
	saver = tf.train.Saver()
	saver.save(m.sess, '/projects/kumar-lab/mehtav/models/' + args.model + '/' + args.model)

	def generate_random_start_state(full_traj):
		random_traj = full_traj[np.random.choice(len(full_traj), 1)[0]]
		start_state_index = np.random.choice(np.arange(N_FRAMES-1, len(random_traj)), 1)[0]
		# print(random_traj[0][0].shape)
		# start_state = np.array(random_traj)
		# print(start_state.shape)
		start_state = random_traj[start_state_index][0]
		start_state = np.ravel(start_state)
		print("Start state: ")
		print(start_state.shape)
		return start_state
	# g = MiceEnv(start_state)
	
	hf = h5py.File('/projects/kumar-lab/mehtav/generated_traj/gt_tail.h5', 'w')
	# generated_traj = {}
	for i in range(m.k):
		g = hf.create_group('option'+str(i))
		for j in range(n_start_states):
			start_state = generate_random_start_state(full_traj)
			print('Visualising option ' + str(i))
			traj = m.visualizePolicy(i, start_state, frames=600, trajectory_only = True, n_frames = N_FRAMES)
			# generated_traj = traj
			g.create_dataset('traj'+str(j), data=np.array(traj))
			# g_ = g.create_group('traj'+str(j))
			# g_.create_dataset('states', data = np.array(traj[:][0]))
			# g_.create_dataset('actions', data = np.array(traj[:][1]))
	hf.close()

	# with open('/projects/kumar-lab/mehtav/generated_traj/gt_tail.pkl', 'wb') as f:
	# 	pickle.dump(generated_traj, f)


def main():
	
	CENTER_SPINE_INDEX = 6
	BASE_TAIL_INDEX = 9
	# Sampling 1000 traj randomly
	directory = '/projects/kumar-lab/mehtav/final_data/'
	# directory = '/gpfs/ctgs0/home/c-mehtav/DDOmice2/temp/'
	filelist = os.listdir(directory)
	# filelist = np.random.choice(filelist, int(args.ntrajs))

	data = []
	for filename in filelist[0]:
		hf = h5py.File(directory+filename, 'r')
		key_list = [key for key in hf.keys()]
		for i in range(int(args.ntrajs)):
			key = np.random.choice(key_list, 1)
			pointtraj = hf.get(key+ '/points')
			conftraj = hf.get(key+ '/confidence')
			velocitytraj = hf.get(key+ '/velocity')
			traj = [pointtraj, conftraj, velocitytraj]
			data.append(traj)

	# data = [file_number, points/conf/vel]
	# Each traj has shape (nrow x 12 x 2) or (nrow by 12)

	count = 0
	full_traj = []
	for file in range(len(data)):
		print(count)
		count = count+1
		# print(data[file][0][0].shape)
		p_traj = data[file][0]
		c_traj = data[file][1]
		v_traj = data[file][2]
		print('state', p_traj[0])
		print('state', p_traj[1])
		print('velocity', v_traj[0],  v_traj[0].mean())
		op_traj = []
		for i in range(N_FRAMES-1, v_traj.shape[0]):
			# print(p_traj.shape)
			s_ = p_traj[i- N_FRAMES+1 : i+1]
			s = np.ravel(s_) # Taking the last 16 frames as state
			a = np.ravel(v_traj[i]) # Removing center spine, as it is always at rest	
			# s = np.delete(s, 17, 0) # Removing x coordinate of base tail
			# a = np.delete(a, 17, 0) # Removing x coordinate of base tail
			op_traj.append((s, a))
		full_traj.append(op_traj)


	# runPolicies(full_traj, np.eye(14), n_start_states = 5) #np.load('/projects/kumar-lab/mehtav/cov.npy').astype(np.float32), file_name = 'LR1e0')


main()


