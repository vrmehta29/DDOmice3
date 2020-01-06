import tensorflow as tf
import numpy as np
from numpy.random import RandomState
from segmentcentroid.inference.forwardbackward import ForwardBackward
from tensorflow.python.client import timeline
from sklearn.cluster import KMeans

class TFModel(object):
	"""
	This class defines the basic data structure for a hierarchical control model. This
	is a wrapper that handles I/O, Checkpointing, and Training Primitives.
	"""

	def __init__(self, 
				 statedim, 
				 actiondim, 
				 k,
				 boundary_conditions,
				 prior,
				 checkpoint_file='/tmp/model.bin',
				 checkpoint_freq=10):
		"""
		Create a TF model from the parameters

		Positional arguments:
		statedim -- numpy.ndarray defining the shape of the state-space
		actiondim -- numpy.ndarray defining the shape of the action-space
		k -- float defining the number of primitives to learn

		Keyword arguments:
		checkpoint_file -- string filname to store the learned model
		checkpoint_freq -- int iter % checkpoint_freq the learned model is checkpointed
		"""

		self.statedim = statedim 
		self.actiondim = actiondim 

		self.k = k

		if not self.restore:
			self.sess = tf.Session()
		self.initialized = False

		self.checkpoint_file = checkpoint_file
		self.checkpoint_freq = checkpoint_freq

		# Creating the list of neural networks
		if self.restore:
			self.initialize_restore()
		else:
			self.initialize()

		self.fb = ForwardBackward(self, boundary_conditions, prior)

		self.saver = tf.train.Saver()

		self.trajectory_cache = {}

		# with tf.variable_scope("optimizer"):
		# 	self.opt = tf.train.AdamOptimizer(learning_rate= self.learning_rate)
		# 	self.loss, self.optimizer, self.init, self.pivars, self.psivars, self.lossa = self.getOptimizationVariables(self.opt)


	def initialize(self):
		"""
		The initialize command is implmented by all subclasses and designed 
		to initialize whatever internal state is needed.
		"""

		raise NotImplemented("Must implement an initialize function")


	def restore(self):
		"""
		Restores the model from the checkpointed file
		"""

		self.saver.restore(self.sess, self.checkpoint_file)

	def save(self):
		"""
		Saves the model to the checkpointed file
		"""

		self.saver.save(self.sess, self.checkpoint_file)

	def evalpi(self, index, traj):
		"""
		Returns the probability of action a at state s for primitive index i

		Positional arguments:
		index -- int index of the required primitive in {0,...,k}
		traj -- a trajectory

		Returns:
		float -- probability
		"""

		if index >= self.k:
			raise ValueError("Primitive index is greater than the number of primitives")

		X, A = self.formatTrajectory(traj)

		return np.abs(self._evalpi(index, X, A))


	def evalpsi(self, index, traj):
		"""
		Returns the probability of action a at state s

		Positional arguments:
		index -- int index of the required primitive in {0,...,k}
		traj -- a trajectory

		Returns:
		float -- probability
		"""

		if index >= self.k:
			raise ValueError("Primitive index is greater than the number of primitives")


		X, _ = self.formatTrajectory(traj)

		return np.abs(self._evalpsi(index, X))


	def _evalpi(self, index, X, A):
		"""
		Sub classes must implment this actual execution routine to eval the probability

		Returns:
		float -- probability
		"""
		raise NotImplemented("Must implement an _evalpi function")


	def _evalpsi(self, index, X):
		"""
		Sub classes must implment this actual execution routine to eval the probability

		Returns:
		float -- probability
		"""
		raise NotImplemented("Must implement an _evalpsi function")

	def getLossFunction(self):
		"""
		Sub classes must implement a function that returns the loss and trainable variables

		Returns:
		loss -- tensorflow function
		pivars -- variables that handle policies
		psivars -- variables that handle transitions
		"""
		raise NotImplemented("Must implement a getLossFunction")


	def dataTransformer(self, trajectory):
		"""
		Sub classes can implement a data augmentation class. The default is the identity transform

		Positional arguments: 
		trajectory -- input is a single trajectory

		Returns:
		trajectory
		"""
		return trajectory


	"""
	####
	Fitting functions. Below we include functions for fitting the models.
	These are mostly for convenience
	####
	"""



	def sampleBatch(self, X):
		"""
		sampleBatch executes the forward backward algorithm and returns
		a single batch of data to train on.

		Positional arguments:
		X -- a list of trajectories. Each trajectory is a list of tuples of states and actions
		dataTransformer -- a data augmentation routine
		"""

		# traj_index = np.random.choice(len(X))
		traj_index = self.batchcounter% self.n_trajs
		import datetime
		now  = datetime.datetime.now()
		trajectory = self.trajectory_cache[traj_index]
		self.batchcounter = self.batchcounter+1

		#print("Time", datetime.datetime.now()-now)

		now  = datetime.datetime.now()
		weights = self.fb.fit([trajectory])


		#Sknote: fetches Q, B from the forward backward algorithm
		Q , B = weights[0][0] , weights[0][1]


		feed_dict = {}
		Xm, Am = self.formatTrajectory(trajectory)


		#prevent stupid shaping errors
		if Xm.shape[0] != weights[0][0].shape[0] or \
		   Am.shape[0] != weights[0][1].shape[0]:
			raise ValueError("Error in shapes in np array passed to TF")


		for j in range(self.k):
			feed_dict[self.pivars[j][0]] = Xm
			feed_dict[self.pivars[j][1]] = Am
			feed_dict[self.pivars[j][2]] = np.reshape(Q[:,j], (Xm.shape[0],1))

			feed_dict[self.psivars[j][0]] = Xm

			#Sknote: format transitions creates a vector [q-b, b]
			feed_dict[self.psivars[j][1]] = self.formatTransitions(Q[:,j], B[:,j])
			
			feed_dict[self.psivars[j][2]] = np.reshape(np.ones((Xm.shape[0],1)), (Xm.shape[0],1))

		return feed_dict


	def PretrainBatch(self, X):
		"""
		Positional arguments:
		X -- a list of trajectories. Each trajectory is a list of tuples of states and actions
		dataTransformer -- a data augmentation routine
		"""
		feed_dict = {}
		for curr_cluster in range(self.k):
				feed_dict[self.pivars[curr_cluster][0]] = []
				feed_dict[self.pivars[curr_cluster][1]] = []
				feed_dict[self.pivars[curr_cluster][2]] = []

		for traj_index in range(len(self.trajectory_cache)):

			trajectory = self.trajectory_cache[traj_index]

			assert len(trajectory[0][0])>2, "Cluster labels might be missing"

			Xm, Am, Lm = self.formatTrajectory(trajectory)

			for t in range(len(trajectory)):
				curr_cluster =  trajectory[t][2] # Label
				feed_dict[self.pivars[curr_cluster][0]].append(Xm[t])
				feed_dict[self.pivars[curr_cluster][1]].append(Am[t])
				feed_dict[self.pivars[curr_cluster][2]].append((1))

		return feed_dict



	def samplePretrainBatch(self, dict):
		"""
		sampleBatch executes the forward backward algorithm and returns
		a single batch of data to train on.

		Positional arguments:
		X -- a list of trajectories. Each trajectory is a list of tuples of states and actions
		dataTransformer -- a data augmentation routine
		"""

		batchsize = 128
		feed_dict = {}

		for curr_cluster in range(self.k):
			nsamples = len(dict[self.pivars[curr_cluster][0]])
			# random_indices =  np.array(np.random.choice(np.arange(nsamples), replace = False, size = batchsize))
			random_index = np.random.choice(nsamples - batchsize, replace = False)
			random_indices = np.arange(random_index, random_index+ batchsize)
			feed_dict[self.pivars[curr_cluster][0]] = np.array(dict[self.pivars[curr_cluster][0]])[random_indices]
			feed_dict[self.pivars[curr_cluster][1]] = np.array(dict[self.pivars[curr_cluster][1]])[random_indices]
			feed_dict[self.pivars[curr_cluster][2]] = np.array(dict[self.pivars[curr_cluster][2]])[random_indices]
			feed_dict[self.pivars[curr_cluster][2]] = np.expand_dims(feed_dict[self.pivars[curr_cluster][2]], axis = 1)

		return feed_dict


		# # traj_index = np.random.choice(len(X))
		# traj_index = self.batchcounter% self.n_trajs
		# trajectory = self.trajectory_cache[traj_index]
		# self.batchcounter = self.batchcounter+1

		# assert len(trajectory[0][0])>2, "Cluster labels might be missing"

		# feed_dict = {}
		# Xm, Am, Lm = self.formatTrajectory(trajectory)

		# for curr_cluster in range(self.k):
		# 	feed_dict[self.pivars[curr_cluster][0]] = []
		# 	feed_dict[self.pivars[curr_cluster][1]] = []
		# 	feed_dict[self.pivars[curr_cluster][2]] = []
		# 	# feed_dict[self.psivars[curr_cluster][0]] = []
		# 	# feed_dict[self.psivars[curr_cluster][1]] = []
		# 	# feed_dict[self.psivars[curr_cluster][2]] = []

		# for t in range(len(trajectory)):
		# 	curr_cluster =  trajectory[t][2] # Label
		# 	feed_dict[self.pivars[curr_cluster][0]].append(Xm[t])
		# 	feed_dict[self.pivars[curr_cluster][1]].append(Am[t])
		# 	feed_dict[self.pivars[curr_cluster][2]].append(1)
		# 	# feed_dict[self.psivars[curr_cluster][0]].append(Xm[t])
		# 	# feed_dict[self.psivars[curr_cluster][1]].append([1,1])
		# 	# feed_dict[self.psivars[curr_cluster][2]].append(1)

		# return feed_dict


	
	def sampleInitializationBatch(self, X, randomBatch, initializationModel):


		#loss, pivars, psivars = self.getLossFunction()

		traj_index = np.random.choice(len(X))

		trajectory = self.trajectory_cache[traj_index]

		weights = (np.zeros((len(X[traj_index]), self.k)), np.ones((len(X[traj_index]), self.k)))

		for i,t in enumerate(trajectory):
			state = t[0].reshape(1,-1)

			index = initializationModel.predict(state)#int(i/ ( len(trajectory)/self.k ))
			weights[0][i, index] = 1
			weights[1][i, index] = 0

		feed_dict = {}
		Xm, Am = self.formatTrajectory(trajectory)

		for j in range(self.k):
			feed_dict[self.pivars[j][0]] = Xm
			feed_dict[self.pivars[j][1]] = Am
			feed_dict[self.pivars[j][2]] = np.reshape(weights[0][:,j], (Xm.shape[0],1))

			feed_dict[self.psivars[j][0]] = Xm
			feed_dict[self.psivars[j][1]] = self.formatTransitions(weights[0][:,j], weights[1][:,j])
			feed_dict[self.psivars[j][2]] = np.reshape(weights[0][:,j], (Xm.shape[0],1))

		return feed_dict

		
	def formatTrajectory(self, 
						 trajectory, 
						 statedim=None, 
						 actiondim=None):
		"""
		Internal method that unzips a trajectory into a state and action tuple

		Positional arguments:
		trajectory -- a list of state and action tuples.
		"""

		#print("###", statedim, actiondim)

		pretrain = len(trajectory[0])>2

		if statedim == None:
			statedim = self.statedim

		if actiondim == None:
			actiondim = self.actiondim

		sarraydims = [s for s in statedim]
		sarraydims.insert(0, len(trajectory))
		#creates an n+1 d array 

		aarraydims = [a for a in actiondim]
		aarraydims.insert(0, len(trajectory))
		#creates an n+1 d array 

		X = np.zeros(tuple(sarraydims))
		A = np.zeros(tuple(aarraydims))
		L = np.zeros((sarraydims[0], 1))

		for t in range(len(trajectory)):
			#print(t, trajectory[t][0], trajectory[t][0].shape, statedim)
			s = np.reshape(trajectory[t][0], statedim)
			a = np.reshape(trajectory[t][1], actiondim)
			if pretrain:
				l = trajectory[t][2]
				L[t, :] = l

			X[t,:] = s
			A[t,:] = a
			

		#special case for 2d arrays
		if len(statedim) == 2 and \
			statedim[1] == 1:
			X = np.squeeze(X,axis=2)
			#print(X.shape)
		
		if len(actiondim) == 2 and \
			actiondim[1] == 1:
			A = np.squeeze(A,axis=2)
			#print(A.shape)

		if pretrain:
			return X, A, L
		else:
			return X, A

	def formatTransitions(self, q, b):
		"""
		Internal method that turns a transition sequence (array of floats [0,1])
		into an encoded array [1-a, a]
		"""

		X = np.zeros((len(q),2))
		for t in range(len(q)):
			X[t,0] = (q[t] - b[t]) 
			X[t,1] = b[t] 
		
		return X


	def getOptimizationVariables(self, opt, pretrain= False):
		"""
		This is an internal method that returns the tensorflow refs
		needed for optimization.

		Positional arguments:
		opt -- a tf.optimizer
		"""

		if pretrain:
			loss, pivars, lossa = self.getPolicyLossFunction()
			psivars = []
		else:
			loss, pivars, psivars, lossa = self.getLossFunction()

		train = opt.minimize(loss)

		list_of_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='optimizer')
		init = tf.initialize_variables(list_of_variables)

		return (loss, train, init, pivars, psivars, lossa)


	def startTraining(self, opt):
		"""
		This method initializes the training routine

		opt -- is the chosen optimizer to use
		"""

		self.gradients = opt.compute_gradients(self.loss)
		self.sess.run(self.init)
		self.initialized = True
		#tf.get_default_graph().finalize()


	def train(self, opt, X, n_trajs, iterations, vqiterations=100, vqbatchsize=25, tb_dir = '1'):
		"""
		This method trains the model on a dataset weighted by the forward 
		backward algorithm

		Positional arguments:
		opt -- a tf.optimizer
		X -- a list of trajectories
		iterations -- the number of iterations
		vqiterations -- the number of iterations to initialize via vq
		"""

		# print('pretrain', len(X[0][0]))
		pretrain = len(X[0][0])>2
		with tf.variable_scope("optimizer"):
			self.opt = tf.train.AdamOptimizer(learning_rate= self.learning_rate)
			self.loss, self.optimizer, self.init, self.pivars, self.psivars, self.lossa = self.getOptimizationVariables(self.opt, pretrain)

		self.n_trajs = n_trajs

		if not self.initialized:

			self.startTraining(opt)

			for i,x in enumerate(X):
				# Trajectory cache is a dictionary
				self.trajectory_cache[i] = self.dataTransformer(x)

			#print("abc")

		if vqiterations != 0:
			self.runVectorQuantization(X, vqiterations, vqbatchsize)
			

		# self.merged = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter(tb_dir)
		self.writer.add_graph(self.sess.graph)

		for option_ in range(self.k):
				# mean_loss = tf.math.reduce_mean(logprob)
				tf.summary.scalar('Policy loss for option ' + str(option_),  tf.math.reduce_mean(self.policy_networks[option_]['wlprob']))
				if not pretrain:
					tf.summary.scalar('Transition loss for option ' + str(option_), tf.math.reduce_mean(self.transition_networks[option_]['wlprob']))

		self.batchcounter = 0

		if pretrain:
			fullBatches = self.PretrainBatch(X)

		for it in range(iterations):

			#if it % self.checkpoint_freq == 0:
				#print("Checkpointing Train", it, self.checkpoint_file)
				#self.save()

			if pretrain:
				batch = self.samplePretrainBatch(fullBatches)
			else:
				batch = self.sampleBatch(X)

			if pretrain:
				print("Iteration", it)
			else:
				print("Iteration", it, np.argmax(self.fb.Q,axis=1))
			print("cLoss 1",  np.mean(self.sess.run(self.policy_networks[0]['wlprob'], batch)))
			print("cLoss 2" , np.mean(self.sess.run(self.policy_networks[1]['wlprob'], batch)))
			if not pretrain:
				print("tLoss 1",np.mean(self.sess.run(self.transition_networks[0]['wlprob'], batch)))
				print("tLoss 2" , np.mean(self.sess.run(self.transition_networks[1]['wlprob'], batch)))

			self.sess.run(self.optimizer, batch)
			self.merged = tf.summary.merge_all()
			s = self.sess.run(self.merged, batch)
			self.writer.add_summary(s, it)

			# assert not self.sess.run(tf.reduce_all(tf.equal(tf.get_variable(initializer = self.sess.run('W1_1:0'), name = "compare1"), 
			# 												tf.get_variable(initializer = self.sess.run('W1_1:0'), name = "compare2")))), "Weights are equal"

			#gradients_materialized = self.sess.run(self.gradients, batch)

		#return gradients_materialized #Assumption [(t, gradients_materialized[i]) for i,t in enumerate(tf.trainable_variables())]



	def runVectorQuantization(self, X, vqiterations, vqbatchsize):
		"""
		This function uses vector quantization to initialize the primitive inference
		"""

		state_action_array = []

		for x in X:
			trajectory = self.dataTransformer(x)
			#print([t[0]  for t in trajectory])

			state_action_array.extend([ np.ravel(t[0].reshape(1, -1))  for t in trajectory])
		

		kmeans = KMeans(n_clusters=self.k, init ='k-means++')
		kmeans.fit(state_action_array)

		"""
		from sklearn.decomposition import PCA
		import matplotlib.pyplot as plt
		p = PCA(n_components=2)
		x = p.fit_transform(state_array)
		plt.scatter(x[:,0], x[:,1])
		plt.show()
		raise ValueError("Break Point")
		"""

		for i in range(vqiterations):
			batch = self.sampleInitializationBatch(X, vqbatchsize, kmeans)
			self.sess.run(self.optimizer, batch)
			print("VQ Loss",self.sess.run(self.loss, batch))
			#print("b",self.sess.run(self.lossa[0], batch))
			#print("b1",self.sess.run(self.lossa[0], batch).shape)
			#print("c",self.sess.run(self.lossa[self.k], batch))



	def visualizePolicy(self, option_no, start_state, frames = 600, filename=None, trajectory_only = True, n_frames = 16, n_frames_after = 4):
			# cmap = colors.ListedColormap(['w', '.75', 'b', 'g', 'r', 'k'], 'GridWorld')

			state = np.reshape(start_state, (1, 14*n_frames))
			# self.sess = tf.Session()

			traj = []
			vis_traj = []
			for i in range(frames):
				feed_dict = {}
				print('Shape: ')
				print(state.shape)
				feed_dict[self.policy_networks[option_no]['state']] = np.reshape(state, (1, 14*n_frames)).astype(np.float32)
				feed_dict[self.policy_networks[option_no]['action']] = np.reshape(np.zeros(14*n_frames_after), (n_frames_after, 14)).astype(np.float32)
				feed_dict[self.policy_networks[option_no]['weight']] = np.reshape(np.ones(1), (1, 1)).astype(np.float32)
				# feed_dict = {'x': [state], 'a': [np.zeros(22)], 'weight': [1]}
				# feed_dict[x] = [state]
				# feed_dict['a:0'] = [np.zeros(22)]
				# feed_dict['weight:0'] = [1]
				a = self.sess.run(self.policy_networks[option_no]['amax'], feed_dict)
				# noise = np.random.uniform(0, 1, (a.shape))
				# a = a + noise

				traj.append((state, a))
				PX_TO_CM = 19.5 * 2.54 / 400

				reshaped_state = np.reshape(state, (n_frames, 14))
				reshaped_a = np.reshape(a, (n_frames_after, 14))
				curr_frame = reshaped_state[-1]
				next_frame = curr_frame + a[0]/30/PX_TO_CM

				print('Action: ')
				print(a.shape, curr_frame.shape)
				vis_traj.append((curr_frame, np.reshape(a[0], (14))))

				next_state = np.empty_like(reshaped_state)
				for i in range(n_frames):
					if i!=n_frames-1:
						next_state[i] = reshaped_state[i+1]
					else:
						next_state[i] = next_frame

				state = np.ravel(next_state)
				print(state.shape)

			if trajectory_only:
				return vis_traj

			if not trajectory_only:
				NOSE = 0
				LEFT_EAR = 1
				RIGHT_EAR = 2
				BASE_NECK = 3
				LEFT_FRONT_PAW = 4
				RIGHT_FRONT_PAW = 5
				# CENTER_SPINE = 6
				LEFT_REAR_PAW = 6
				RIGHT_REAR_PAW = 7
				BASE_TAIL = 8
				MID_TAIL = 9
				TIP_TAIL = 10   

				plt.style.use('seaborn-pastel')


				fig = plt.figure()
				ax = plt.axes(xlim=(-50, 100), ylim=(-50, 50))
				line, = ax.plot([], [], lw=3)

				def init():
					line.set_data([], [])
					return line,

				def rebuild_state(s):
					# s = np.reshape(s, (21))
					# Adding the x coordinate for base tail
					# s = np.insert(s, 17, 0)
					return s

				def animate(i):
					s = rebuild_state(traj[i][0])
					s = s.reshape((11,2))
				#   # a = traj[1].reshape((11,2))
					x = s[:, 0]
					y = s[:, 1]
					plt_x = np.array([  x[RIGHT_EAR], x[LEFT_EAR], x[NOSE], x[RIGHT_EAR],  x[BASE_NECK], 0, x[RIGHT_FRONT_PAW], 0, x[LEFT_FRONT_PAW], 
										0, x[BASE_TAIL], x[RIGHT_REAR_PAW], x[BASE_TAIL], x[LEFT_REAR_PAW], x[BASE_TAIL], x[MID_TAIL], x[TIP_TAIL]  ])
					plt_y = np.array([  y[RIGHT_EAR], y[LEFT_EAR], y[NOSE], y[RIGHT_EAR],  y[BASE_NECK], 0, y[RIGHT_FRONT_PAW], 0, y[LEFT_FRONT_PAW], 
										0, y[BASE_TAIL], y[RIGHT_REAR_PAW], y[BASE_TAIL], y[LEFT_REAR_PAW], y[BASE_TAIL], y[MID_TAIL], y[TIP_TAIL]       ])
					line.set_data(plt_x, plt_y)
					return line,

				anim = FuncAnimation(fig, animate, init_func=init, frames=len(traj), interval=100/3, blit=True)

				if filename == None:
					plt.show()
				else:
					plt.savefig(filename)


		
