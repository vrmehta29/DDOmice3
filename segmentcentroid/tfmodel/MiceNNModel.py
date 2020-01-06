from .TFSeparableModel import TFSeparableModel
from .supervised_networks import *
import numpy as np
import os

class MiceNNModel(TFSeparableModel):
	
	"""
	This class defines the abstract class for a tensorflow model for the primitives.
	"""

	def __init__(self, 
				 k,
				 statedim=(14,1), 
				 actiondim=(14,1), 
				 hidden_layer=[48],
				 cov_mat = np.eye(14),
				 init_weights = 'normal',
				 learning_rate = 0.001,
				 restore= False,
				 model_dir = '',
				 model_name = '',

				 ):

		self.hidden_layer = hidden_layer
		self.statedim = statedim
		self.actiondim = actiondim
		self.cov_mat = cov_mat
		self.MLPcounter = 0
		self.MLPcounter2 = 0
		self.shared_params = []
		self.shared_params2 = []
		# Supported initializations: ['xavier', 'normal']
		self.init_weights = init_weights 
		self.learning_rate = learning_rate
		self.restore = restore

		if restore:
			self.sess = tf.Session()
			print('mnn  ',model_dir, model_name)
			new_saver = tf.train.import_meta_graph(os.path.join(model_dir, model_name) + '.meta')
			new_saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))


		super(MiceNNModel, self).__init__(statedim, actiondim, k, [0,1], 'chain')


	def createPolicyNetwork(self):

		self.MLPcounter = self.MLPcounter+ 1
		network, self.shared_params = MLPcont(self.statedim[0], self.actiondim[0], self.hidden_layer, self.cov_mat, self.MLPcounter, 
						self.init_weights, self.shared_params, 'relu', self.k)
		return network
		#return gridWorldTabular(10, 20, 4)

	def createTransitionNetwork(self):

		self.MLPcounter2 = self.MLPcounter2 +1
		network, self.shared_params2 = multiLayerPerceptron(self.statedim[0], 2, hidden_layer = self.hidden_layer, 
				MLPcounter = self.MLPcounter2, init_weights= self.init_weights, shared_params = self.shared_params2)
		return network


	def restore_network(self):
		"""
		This function creates a classification network that takes states and
		predicts a hot-one encoded action. It is based on a MLP.

		Positional arguments:
		sdim -- int dimensionality of the state-space
		adim -- int dimensionality of the action-space
		
		Keyword arguments:
		hidden_later -- int size of the hidden layer
		
		"""
		sdim = self.statedim[0]
		adim = self.actiondim[0]
		hidden_layer = self.hidden_layer
		cov_mat = self.cov_mat
		self.MLPcounter = self.MLPcounter+ 1

		x = tf.placeholder(tf.float32, shape=[None, sdim])

		#must be one-hot encoded
		a = tf.placeholder(tf.float32, shape=[None, adim])

		#must be a scalar
		weight = tf.placeholder(tf.float32, shape=[None, 1])


		W = []
		b = []
		h = [x]
		prev_layer_size = sdim
		for layer_number, layer_size in enumerate(hidden_layer):
			layer_number = layer_number+ 1 # Starting from 1 instead of 0

			if layer_number==1:
				W.append(tf.get_variable(initializer = self.sess.run('W' + str(layer_number)+ '_'+ str(1) + ':0'), name = 'rW' + str(layer_number)+ '_' + str(self.MLPcounter)))
				b.append(tf.get_variable(initializer = self.sess.run('b' + str(layer_number)+ '_'+ str(1) + ':0'), name = 'rb' + str(layer_number)+ '_' + str(self.MLPcounter)))
			else:
				W.append(tf.get_variable(initializer = self.sess.run('W' + str(layer_number)+ '_'+ str(self.MLPcounter) + ':0'), name = 'rW' + str(layer_number)+ '_' + str(self.MLPcounter)))
				b.append(tf.get_variable(initializer = self.sess.run('b' + str(layer_number)+ '_'+ str(self.MLPcounter) + ':0'), name = 'rb' + str(layer_number)+ '_' + str(self.MLPcounter)))
			h.append(tf.nn.relu(tf.matmul(h[-1], W[layer_number-1]) + b[layer_number-1]))
			prev_layer_size = layer_size

		#h1 = tf.nn.dropout(h1, 0.5)

		W_out = tf.get_variable(initializer = self.sess.run('Wout_'+ str(self.MLPcounter) + ':0'), name = 'rWout_' + str(self.MLPcounter))
		b_out = tf.get_variable(initializer = self.sess.run('bout_'+ str(self.MLPcounter) + ':0'), name = 'rbout_' + str(self.MLPcounter))
			
		mean = tf.matmul(h[-1], W_out) + b_out
		
		dist = tfd.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=cov_mat.astype(np.float32))
		dist2 = tfd.Normal(loc=mean, scale=cov_mat.astype(np.float32)[0,0])
	
		# Multiplying pdf with a small value compared to the observations for calculating probability
		# pdf = dist.prob(a)
		pdf2 = dist2.prob(a)
		y = tf.clip_by_value(pdf2, clip_value_min = 10**-37, clip_value_max = 1.0)
		y_prob = tf.math.reduce_prod(y, axis = 1)

		#Sknote: a1 log(pred) - a2 log(1-pred)
		logprob = tf.reduce_mean(tf.math.scalar_mul(-1, tf.reduce_sum(tf.math.log(y), axis=1)))
		
		# mean_loss = tf.math.reduce_mean(logprob)
		# tf.summary.scalar('Policy loss for option ' + str(MLPcounter), mean_loss)

		wlogprob = tf.multiply(weight, logprob)
			
		return {'state': x, 
					'action': a, 
					'weight': weight,
					'prob': y_prob, 
					'amax': mean,
					'lprob': logprob,
					'wlprob': wlogprob,
					'discrete': False}


	def restore_trans_network(self):
		"""
		This function creates a classification network that takes states and
		predicts a hot-one encoded action. It is based on a MLP.

		Positional arguments:
		sdim -- int dimensionality of the state-space
		adim -- int dimensionality of the action-space
		
		Keyword arguments:
		hidden_later -- int size of the hidden layer
		
		"""
		sdim = self.statedim[0]
		adim = 2
		hidden_layer = self.hidden_layer
		cov_mat = self.cov_mat
		self.MLPcounter2 = self.MLPcounter2 +1

		x = tf.placeholder(tf.float32, shape=[None, sdim])

		#must be one-hot encoded
		a = tf.placeholder(tf.float32, shape=[None, adim])

		#must be a scalar
		weight = tf.placeholder(tf.float32, shape=[None, 1])


		W = []
		b = []
		h = [x]
		prev_layer_size = sdim
		for layer_number, layer_size in enumerate(hidden_layer):
			layer_number = layer_number+ 1 # Starting from 1 instead of 0

			if layer_number==1:
				W.append(tf.get_variable(initializer = self.sess.run('trans_W' + str(layer_number)+ '_'+ str(1) + ':0'), name = 'trans_rW' + str(layer_number)+ '_' + str(self.MLPcounter2)))
				b.append(tf.get_variable(initializer = self.sess.run('trans_b' + str(layer_number)+ '_'+ str(1) + ':0'), name = 'trans_rb' + str(layer_number)+ '_' + str(self.MLPcounter2)))
			else:
				W.append(tf.get_variable(initializer = self.sess.run('trans_W' + str(layer_number)+ '_'+ str(self.MLPcounter2) + ':0'), name = 'trans_rW' + str(layer_number)+ '_' + str(self.MLPcounter2)))
				b.append(tf.get_variable(initializer = self.sess.run('trans_b' + str(layer_number)+ '_'+ str(self.MLPcounter2) + ':0'), name = 'trans_rb' + str(layer_number)+ '_' + str(self.MLPcounter2)))
			h.append(tf.nn.sigmoid(tf.matmul(h[-1], W[layer_number-1]) + b[layer_number-1]))
			prev_layer_size = layer_size

		#h1 = tf.nn.dropout(h1, 0.5)

		W_out = tf.get_variable(initializer = self.sess.run('trans_Wout_'+ str(self.MLPcounter2) + ':0'), name = 'trans_rWout_' + str(self.MLPcounter2))
		b_out = tf.get_variable(initializer = self.sess.run('trans_bout_'+ str(self.MLPcounter2) + ':0'), name = 'trans_rbout_' + str(self.MLPcounter2))
			
		logit = tf.matmul(h[-1], W_out) + b_out
		y = tf.nn.softmax(logit)

		#Sknote: a1 log(pred) - a2 log(1-pred)
		logprob = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits = logit, labels = a), [-1,1])

		# mean_loss = tf.math.reduce_mean(logprob)
		# tf.summary.scalar('Transition loss for option ' + str(MLPcounter), mean_loss)

		wlogprob = tf.multiply(weight, logprob)
			
		return {'state': x, 
					'action': a, 
					'weight': weight,
					'prob': y, 
					'amax': tf.argmax(y, 1),
					'lprob': logprob,
					'wlprob': wlogprob,
					'discrete': True}

