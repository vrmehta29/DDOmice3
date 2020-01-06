import matplotlib.pyplot as plt
import numpy as np
import pickle
import h5py
import os
import numpy as np
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import argparse
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import pickle
from segmentcentroid.tfmodel.MiceNNModel import MiceNNModel
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--to_label')
parser.add_argument('--model_dir')
parser.add_argument('--model_name')
parser.add_argument('--hidden_layer')
parser.add_argument('--number_options')
parser.add_argument('--n_trajs')
# parser.add_argument('--save_video')
args = parser.parse_args()
# option_no = args.option_no

N_FRAMES = 16
N_FRAMES_AFTER = 4


hidden_layer = [int(i) for i in args.hidden_layer.split(',')]
m  = MiceNNModel(int(args.number_options), statedim=(14*N_FRAMES, 1), actiondim = (14*N_FRAMES_AFTER, 1)
			, hidden_layer = hidden_layer, cov_mat = np.eye(14* N_FRAMES_AFTER), restore= True, model_dir = args.model_dir, model_name = args.model_name)

policy_networks = []
transition_networks = []

# def initialize(sdim, adim):
# 		"""
# 		Initializes the internal state
# 		"""

# 		for i in range(0, int(args.number_options)):
# 			print('Restoring policy network ', i)
# 			policy_networks.append(m.restore_network(i))

# 		for i in range(0, int(args.number_options)):
# 			print('Restoring transition network ', i)
# 			transition_networks.append(m.restore_trans_network(i))

# 		# for i in range(0, self.k):
# 		#     self.transition_networks.append(restore_network())



def visualizePolicy(traj, frames = 60, filename=None):

	# import seaborn as sns
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
		reshaped_state = np.reshape(state, (N_FRAMES, 14))
		s = reshaped_state[-1]
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
	# plt.style.use('seaborn-pastel')

	fig = plt.figure()
	ax = plt.axes(xlim=(-100, 100), ylim=(-100, 100))
	line, = ax.plot([], [], lw=3)
	scat = ax.scatter([], [], c = 'red')

	anim = FuncAnimation(fig, animate, init_func=init,
								   frames=frames, interval=100/3, blit=True)

	if filename == None:
		plt.show()
	else:
		anim.save(filename + '.mp4', writer = '/projects/kumar-lab/PipelineEnvironment/ffmpeg/ffmpeg-3.3.3-64bit-static/ffmpeg')



# def generate_random_start_state():
# 	hf = h5py.File(args.iptraj_file, 'r')
# 	key_list = [key for key in hf.keys()]
# 	key = np.random.choice(key_list, 1)[0]
# 	pointtraj = hf.get(key+ '/points')[()]
# 	start_state_index = np.random.choice(np.arange(N_FRAMES-1, len(pointtraj)), 1)[0]
# 	# start_state = pointtraj[start_state_index]
# 	start_state = pointtraj[start_state_index- N_FRAMES+1 : start_state_index+1]
# 	start_state = np.ravel(start_state)
# 	print("Start state: ")
# 	print(start_state.shape)
# 	return start_state



def gen_full_traj(read_file, n_trajs):
	data = []
	# for filename in filelist[0]:
	hf = h5py.File(read_file, 'r')
	key_list = [key for key in hf.keys()]
	for i in range(int(n_trajs)):
		print('Reading ' + str(i))
		# key = np.random.choice(key_list, 1)[0]
		key = key_list[i]
		pointtraj = hf.get(key+ '/points')[()]
		conftraj = hf.get(key+ '/confidence')[()]
		velocitytraj = hf.get(key+ '/velocity')[()]
		traj = [pointtraj, conftraj, velocitytraj]
		data.append(traj)

	# data = [file_number, points/conf/vel]
	# Each traj has shape (nrow x 12 x 2) or (nrow by 12)

	count = 0
	full_traj = []
	for file in range(len(data)):
		print(count)
		count = count+1
		p_traj = data[file][0]
		c_traj = data[file][1]
		v_traj = data[file][2]
		op_traj = []
		for i in range(N_FRAMES-1, v_traj.shape[0] - N_FRAMES_AFTER):
			# print(p_traj.shape)
			s_ = p_traj[i- N_FRAMES+1 : i+1]
			s = np.ravel(s_) # Taking the last 16 frames as state
			a = np.ravel(v_traj[i: i+ N_FRAMES_AFTER]) # Removing center spine, as it is always at rest	
			op_traj.append((s, a))
		full_traj.append(op_traj)

	return full_traj


def main():

	# initialize(sdim = 14*N_FRAMES, adim = 14*N_FRAMES_AFTER)
	init_op = tf.global_variables_initializer()
	m.sess.run(init_op)
	print('init')
	# traj_ = visualizePolicy(int(args.option_no), start_state, trajectory_only = True)
	
	X = gen_full_traj(args.to_label, int(args.n_trajs))
	m.n_trajs = int(args.n_trajs)
	for i,x in enumerate(X):
		# Trajectory cache is a dictionary
		m.trajectory_cache[i] = x
	print('gen')
	
	Q_list = []
	m.batchcounter = 0
	for i in range(int(args.n_trajs)):
		batch = m.sampleBatch(X)
		Q_list.append(np.argmax(m.fb.Q,axis=1))
	print('Q')
	print('Q_list', len(Q_list[0]))

	traj_count = np.zeros(m.k)
	for i in range(int(args.n_trajs)):
		curr_option = scipy.stats.mode(Q_list[i])[0][0]
		traj_count[curr_option] = traj_count[curr_option]+ 1
		visualizePolicy(m.trajectory_cache[i], 60, args.model_dir + '/labelled/option_' + str(curr_option) + '_traj_' + str(traj_count[curr_option]))
		print('Visualising traj ' + str(i))		
		# groups[Q_list[i]].create_dataset('traj'+str(j), data=np.array(traj))
	# hf.close()

main()