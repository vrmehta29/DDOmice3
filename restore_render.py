import matplotlib.pyplot as plt
import numpy as np
import pickle
import h5py
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import argparse
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--iptraj_file')
parser.add_argument('--model')
parser.add_argument('--hidden_layer')
parser.add_argument('--number_options')
args = parser.parse_args()
# option_no = args.option_no

N_FRAMES = 16

policy_networks = []
sess = tf.Session()
new_saver = tf.train.import_meta_graph(args.model)
new_saver.restore(sess, tf.train.latest_checkpoint('./'))


def initialize(sdim, adim):
		"""
		Initializes the internal state
		"""
		for i in range(0, int(args.number_options)):
			policy_networks.append(restore_network(sdim, adim, MLPcounter= i))

		# for i in range(0, self.k):
		#     self.transition_networks.append(restore_network())

def restore_network(sdim, 
			 adim, 
			 hidden_layer=[512,512],
			 cov_mat = np.eye(14),
			 MLPcounter = 0):
	"""
	This function creates a classification network that takes states and
	predicts a hot-one encoded action. It is based on a MLP.

	Positional arguments:
	sdim -- int dimensionality of the state-space
	adim -- int dimensionality of the action-space
	
	Keyword arguments:
	hidden_later -- int size of the hidden layer
	
	"""

	hidden_layer = [int(i) for i in args.hidden_layer.split(',')]

	x = tf.placeholder(tf.float32, shape=[None, sdim])

	#must be one-hot encoded
	a = tf.placeholder(tf.float32, shape=[None, adim])

	#must be a scalar
	weight = tf.placeholder(tf.float32, shape=[None, 1])

	# W_h1 = tf.get_variable(initializer = sess.run('W1_'+ str(MLPcounter) + ':0'), name = 'rW1_' + str(MLPcounter))
	# b_1 = tf.get_variable(initializer = sess.run('b1_'+ str(MLPcounter) + ':0'), name = 'rb1_' + str(MLPcounter))
	# h1 = tf.nn.sigmoid(tf.matmul(x, W_h1) + b_1)
	# W_h2 = tf.get_variable(initializer = sess.run('W2_'+ str(MLPcounter) + ':0'), name = 'rW2_' + str(MLPcounter))
	# b_2 = tf.get_variable(initializer = sess.run('b2_'+ str(MLPcounter) + ':0'), name = 'rb2_' + str(MLPcounter))
	# h2 = tf.nn.sigmoid(tf.matmul(h1, W_h2) + b_2)

	W = []
	b = []
	h = [x]
	prev_layer_size = sdim
	for layer_number, layer_size in enumerate(hidden_layer):
		layer_number = layer_number+ 1 # Starting from 1 instead of 0
		W.append(tf.get_variable(initializer = sess.run('W' + str(layer_number)+ '_'+ str(MLPcounter) + ':0'), name = 'rW' + str(layer_number)+ '_' + str(MLPcounter)))
		b.append(tf.get_variable(initializer = sess.run('b' + str(layer_number)+ '_'+ str(MLPcounter) + ':0'), name = 'rb' + str(layer_number)+ '_' + str(MLPcounter)))
		h.append(tf.nn.sigmoid(tf.matmul(h[-1], W[layer_number-1]) + b[layer_number-1]))
		prev_layer_size = layer_size

	#h1 = tf.nn.dropout(h1, 0.5)

	W_out = tf.get_variable(initializer = sess.run('Wout_'+ str(MLPcounter) + ':0'), name = 'rWout_' + str(MLPcounter))
	b_out = tf.get_variable(initializer = sess.run('bout_'+ str(MLPcounter) + ':0'), name = 'rbout_' + str(MLPcounter))
		
	mean = tf.matmul(h[-1], W_out) + b_out
	
	dist = tfd.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=cov_mat.astype(np.float32))
	
	# Multiplying pdf with a small value compared to the observations for calculating probability
	pdf = dist.prob(a)
	y = tf.math.scalar_mul(0.1, pdf)

	#Sknote: a1 log(pred) - a2 log(1-pred)
	logprob = tf.reduce_mean(tf.multiply(tf.math.subtract(mean, a), tf.math.subtract(mean, a)))

	wlogprob = tf.multiply(weight, logprob)
		
	return {'state': x, 
				'action': a, 
				'weight': weight,
				'prob': y, 
				'amax': mean,
				'lprob': logprob,
				'wlprob': wlogprob,
				'discrete': False}

def visualizePolicy(option_no, start_state, frames = 600, filename=None, trajectory_only = False):
		# cmap = colors.ListedColormap(['w', '.75', 'b', 'g', 'r', 'k'], 'GridWorld')

	state = np.reshape(start_state, (1, 14*N_FRAMES))
	# self.sess = tf.Session()

	traj = []
	vis_traj = []
	for i in range(frames):
		feed_dict = {}
		print('Shape: ')
		print(state.shape)
		feed_dict[policy_networks[option_no]['state']] = np.reshape(state, (1, 14*N_FRAMES)).astype(np.float32)
		feed_dict[policy_networks[option_no]['action']] = np.reshape(np.zeros(14), (1, 14)).astype(np.float32)
		feed_dict[policy_networks[option_no]['weight']] = np.reshape(np.ones(1), (1, 1)).astype(np.float32)
		# feed_dict = {'x': [state], 'a': [np.zeros(22)], 'weight': [1]}
		# feed_dict[x] = [state]
		# feed_dict['a:0'] = [np.zeros(22)]
		# feed_dict['weight:0'] = [1]
		a = sess.run(policy_networks[option_no]['amax'], feed_dict)
		noise = np.random.uniform(0, 1, (a.shape))
		a = a + noise

		traj.append((state, a))
		PX_TO_CM = 19.5 * 2.54 / 400

		reshaped_state = np.reshape(state, (N_FRAMES, 14))
		curr_frame = reshaped_state[-1]
		next_frame = curr_frame + a/30/PX_TO_CM

		print('Action: ')
		print(a.shape, curr_frame.shape)
		vis_traj.append((curr_frame, np.reshape(a, (14))))

		next_state = np.empty_like(reshaped_state)
		for i in range(N_FRAMES):
			if i!=N_FRAMES-1:
				next_state[i] = reshaped_state[i+1]
			else:
				next_state[i] = next_frame

		state = np.ravel(next_state)
		print(state.shape)

	if trajectory_only:
		return vis_traj
		
	if not trajectory_only:
		import seaborn as sns
		HEAD = 0
		BASE_NECK = 1
		CENTER_SPINE_INDEX = 2
		LEFT_REAR_PAW = 3
		RIGHT_REAR_PAW = 4
		# BASE_TAIL = 5
		MID_TAIL = 5
		TIP_TAIL = 6

		def init():
			line.set_data([], [])
			scat.set_offsets([])
			return line, scat

		def rebuild_state(s):
			# s = np.reshape(s, (21))
			# Adding the x coordinate for base tail
			# s = np.insert(s, 17, 0)	
			return s

		def animate(i):
			s = rebuild_state(traj[i][0])
			s = s.reshape((7,2))
		# 	# a = traj[1].reshape((11,2))
			x = s[:, 0]
			y = s[:, 1]
			plt_x = np.array([  x[HEAD],  x[BASE_NECK], x[CENTER_SPINE_INDEX], 0, x[RIGHT_REAR_PAW], 0, x[LEFT_REAR_PAW], 0, x[MID_TAIL], x[TIP_TAIL]  ])
			plt_y = np.array([  y[HEAD],  y[BASE_NECK], y[CENTER_SPINE_INDEX], 0, y[RIGHT_REAR_PAW], 0, y[LEFT_REAR_PAW], 0, y[MID_TAIL], y[TIP_TAIL]  ])
			line.set_data(plt_x, plt_y)
			scat.set_offsets(np.array([plt_x, plt_y]).T)
			return scat, line, 	


		# traj = full_traj[option_no]
		plt.style.use('seaborn-pastel')

		# Plotting initial state
		fig1 = plt.figure()
		ax = plt.axes(xlim=(-100, 100), ylim=(-100, 100))
		s0 = rebuild_state(traj[0][0]).reshape((7, 2))
		x = s0[:, 0]
		y = s0[:, 1]
		plt_x = np.array([  x[HEAD],  x[BASE_NECK], x[CENTER_SPINE_INDEX], 0, x[RIGHT_REAR_PAW], 0, x[LEFT_REAR_PAW], 0, x[MID_TAIL], x[TIP_TAIL]  ])
		plt_y = np.array([  y[HEAD],  y[BASE_NECK], y[CENTER_SPINE_INDEX], 0, y[RIGHT_REAR_PAW], 0, y[LEFT_REAR_PAW], 0, y[MID_TAIL], y[TIP_TAIL]  ])
		plt.scatter(plt_x, plt_y, c= 'red')
		plt.plot(plt_x, plt_y)
		fig1.show()

		fig = plt.figure()
		ax = plt.axes(xlim=(-100, 100), ylim=(-100, 100))
		line, = ax.plot([], [], lw=3)
		scat = ax.scatter([], [], c = 'red')

		anim = FuncAnimation(fig, animate, init_func=init,
									   frames=60, interval=100/3, blit=True)

		if filename == None:
			plt.show()
		else:
			anim.save('render.gif', writer='imagemagick')



# def generate_random_start_state():
# 	# random_traj = full_traj[np.random.choice(len(full_traj), 1)[0]]
# 	hf = h5py.File(args.iptraj_file, 'r')
# 	pointtraj = hf.get('points')
# 	start_state = pointtraj[np.random.choice(len(pointtraj), 1)[0]]
# 	print("Start state: ")
# 	print(start_state.shape)
# 	return np.reshape(start_state, (start_state.shape[0]*2))


def generate_random_start_state():
	hf = h5py.File(args.iptraj_file, 'r')
	pointtraj = hf.get('points')
	start_state_index = np.random.choice(np.arange(N_FRAMES-1, len(pointtraj)), 1)[0]
	# start_state = pointtraj[start_state_index]
	start_state = pointtraj[start_state_index- N_FRAMES+1 : start_state_index+1]
	start_state = np.ravel(start_state)
	print("Start state: ")
	print(start_state.shape)
	return start_state



def main():
	start_state = generate_random_start_state()
	initialize(sdim = 14*N_FRAMES, adim = 14)
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	# traj_ = visualizePolicy(int(args.option_no), start_state, trajectory_only = True)
	
	hf = h5py.File('/projects/kumar-lab/mehtav/models/gen_traj/gt_tail.h5', 'w')
	# generated_traj = {}
	for i in range(int(args.number_options)):
		g = hf.create_group('option'+str(i))
		for j in range(5):
			start_state = generate_random_start_state()
			print('Visualising option ' + str(i))
			traj = visualizePolicy(i, start_state, trajectory_only = True)			
			# generated_traj = traj
			g.create_dataset('traj'+str(j), data=np.array(traj))
	hf.close()

main()