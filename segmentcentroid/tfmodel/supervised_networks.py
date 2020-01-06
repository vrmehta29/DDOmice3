import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import random
import string
import tensorflow.contrib.slim as slim

"""
This file is meant to be a model-zoo like file that creates 
networks that can be used in other parts of the pipeline. These
networks are tensorflow references and need to expose the following API.

{ 
'state': <reference to the state placeholder>
'action': <reference to the action placeholder>
'weight': <reference to the weight placeholder>
'prob': <reference to a variable that calculates the probability of action given state>
'wlprob': <reference to a variable that calculates the weighted log probability> 
'discrete': <True/False describing whether the action space should be one-hot encoded>   
}
"""


def continuousTwoLayerReLU(sdim, adim, variance, hidden_layer=64):
	"""
	This function creates a regression network that takes states and
	regresses to actions. It is based on a gated relu.

	Positional arguments:
	sdim -- int dimensionality of the state-space
	adim -- int dimensionality of the action-space
	variance -- float scaling for the probability calculation
	
	Keyword arguments:
	hidden_later -- int size of the hidden layer
	"""

	x = tf.placeholder(tf.float32, shape=[None, sdim])

	a = tf.placeholder(tf.float32, shape=[None, adim])

	weight = tf.placeholder(tf.float32, shape=[None, 1])

	W_h1 = tf.Variable(tf.random_normal([sdim, hidden_layer ]))

	b_1 = tf.Variable(tf.random_normal([hidden_layer]))

	h1 = tf.concat(1,[tf.nn.relu(tf.matmul(x, W_h1) + b_1), tf.matmul(x, W_h1) + b_1])

	W_out = tf.Variable(tf.random_normal([hidden_layer*2, adim]))

	b_out = tf.Variable(tf.random_normal([adim]))

	output = tf.matmul(h1, W_out) + b_out

	logprob = tf.reduce_sum((output-a)**2, 1)/variance

	y = tf.exp(-logprob)

	wlogprob = tf.multiply(tf.transpose(weight), logprob)
		
	return {'state': x, 
				'action': a, 
				'weight': weight,
				'prob': y, 
				'amax': output,
				'lprob': logprob,
				'wlprob': wlogprob,
				'discrete': False}



def logisticRegression(sdim, adim):

	"""
	This function creates a multinomial logistic regression

	Positional arguments:
	sdim -- int dimensionality of the state-space
	adim -- int dimensionality of the action-space
	"""

	x = tf.placeholder(tf.float32, shape=[None, sdim])

	a = tf.placeholder(tf.float32, shape=[None, adim])

	weight = tf.placeholder(tf.float32, shape=[None, 1])

	W_h1 = tf.Variable(tf.random_normal([sdim, adim]))
	b_1 = tf.Variable(tf.random_normal([adim]))
		
	logit = tf.matmul(x, W_h1) + b_1

	y = tf.nn.softmax(logit)

	logprob = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logit, a), [-1,1])

	wlogprob = tf.multiply(weight, logprob)
		
	return {'state': x, 
				'action': a, 
				'weight': weight,
				'prob': y, 
				'amax': tf.argmax(y, 1),
				'lprob': logprob,
				'wlprob': wlogprob,
				'discrete': True}


def multiLayerPerceptron(sdim, 
						 adim, 
						 hidden_layer=[512],
						 MLPcounter= 0,
						init_weights = 'xavier',
						shared_params = [],
						activation = 'sigmoid',
						):
	"""
	This function creates a classification network that takes states and
	predicts a hot-one encoded action. It is based on a MLP.

	Positional arguments:
	sdim -- int dimensionality of the state-space
	adim -- int dimensionality of the action-space
	
	Keyword arguments:
	hidden_later -- int size of the hidden layer
	"""

	if activation == 'sigmoid':
		activation_fn =  tf.nn.sigmoid
	if activation == 'relu':
		activation_fn = tf.nn.relu

	x = tf.placeholder(tf.float32, shape=[None, sdim])

	#must be one-hot encoded
	a = tf.placeholder(tf.float32, shape=[None, adim])

	#must be a scalar
	weight = tf.placeholder(tf.float32, shape=[None, 1])

	def initializer_fn():
		if init_weights == 'xavier':
			return tf.contrib.layers.xavier_initializer(uniform = False)
		if init_weights == 'normal':
			return tf.random_normal_initializer()

	W = []
	b = []
	h = [x]

	with tf.name_scope('Trans_Weights'):
		if MLPcounter==1:
			shared_W = tf.get_variable('trans_W' + str(1)+ '_' + str(MLPcounter), shape=[sdim, hidden_layer[0]], initializer=initializer_fn())
			shared_b = tf.get_variable('trans_b' + str(1)+ '_' + str(MLPcounter), shape=[hidden_layer[0]], initializer=initializer_fn())
			shared_params = [shared_W, shared_b]
		else:
			shared_W = shared_params[0]
			shared_b = shared_params[1]
		W.append(shared_W)
		b.append(shared_b)
		h.append(activation_fn(tf.matmul(h[-1], W[0]) + b[0]))
		prev_layer_size = hidden_layer[0]

		for layer_number, layer_size in enumerate(hidden_layer[1:], start = 1):
			layer_number = layer_number+ 1 # Starting from 1 instead of 0
			# W.append(tf.Variable(tf.random_normal([prev_layer_size, layer_size], stddev = 0.01), name = 'trans_W' + str(layer_number)+ '_' + str(MLPcounter)))
			# b.append(tf.Variable(tf.random_normal([layer_size]), name = 'trans_b' + str(layer_number)+ '_' + str(MLPcounter)))
			# h.append(tf.nn.sigmoid(tf.matmul(h[-1], W[layer_number-1]) + b[layer_number-1]))

			W.append(tf.get_variable('trans_W' + str(layer_number)+ '_' + str(MLPcounter), shape=[prev_layer_size, layer_size], initializer=initializer_fn()))
			b.append(tf.get_variable('trans_b' + str(layer_number)+ '_' + str(MLPcounter), shape=[layer_size], initializer=initializer_fn()))
			h.append(activation_fn(tf.matmul(h[-1], W[layer_number-1]) + b[layer_number-1]))
			prev_layer_size = layer_size


		#h1 = tf.nn.dropout(h1, 0.5)
		tf.summary.scalar('Weights for option ' + str(MLPcounter), tf.math.reduce_mean(W[0]))

		# W_out = tf.get_variable(tf.random_normal([hidden_layer[-1], adim]), name = 'trans_Wout_' + str(MLPcounter))
		# b_out = tf.get_variable(tf.random_normal([adim]), name = 'trans_bout_' + str(MLPcounter))
		W_out = tf.get_variable('trans_Wout_' + str(MLPcounter), shape=[hidden_layer[-1], adim], initializer=initializer_fn())
		b_out = tf.get_variable('trans_bout_' + str(MLPcounter), shape=[adim], initializer=initializer_fn())
		
	logit = tf.matmul(h[-1], W_out) + b_out
	y = tf.nn.softmax(logit)
	# print("TN y: ", tf.shape(y))

	#Sknote: a1 log(pred) - a2 log(1-pred)
	logprob = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(logits = logit, labels = a), [-1,1])

	# mean_loss = tf.math.reduce_mean(logprob)
	# tf.summary.scalar('Transition loss for option ' + str(MLPcounter), mean_loss)

	wlogprob = tf.multiply(weight, logprob)
	print("TN wlog: ", tf.shape(weight))
		
	return {'state': x, 
				'action': a, 
				'weight': weight,
				'prob': y, 
				'amax': tf.argmax(y, 1),
				'lprob': logprob,
				'wlprob': wlogprob,
				'discrete': True}, shared_params

def MLPcont(sdim, 
			 adim, 
			 hidden_layer=[128],
			 cov_mat = np.eye(14),
			 MLPcounter = 0,
			 init_weights = 'xavier',
			 shared_params = [],
			 activation = 'relu',
			 k = 5,
			 stddev = 1):
	"""
	This function creates a classification network that takes states and
	predicts a hot-one encoded action. It is based on a MLP.
	Positional arguments:
	sdim -- int dimensionality of the state-space
	adim -- int dimensionality of the action-space
	
	Keyword arguments:
	hidden_later -- int size of the hidden layer
	
	"""

	if activation == 'sigmoid':
		activation_fn =  tf.nn.sigmoid
	if activation == 'relu':
		activation_fn = tf.nn.relu
	if activation == 'tanh':
		activation_fn = tf.nn.tanh

	x = tf.placeholder(tf.float32, shape=[None, sdim])

	#must be one-hot encoded
	a = tf.placeholder(tf.float32, shape=[None, adim])

	#must be a scalar
	weight = tf.placeholder(tf.float32, shape=[None, 1])
	tf.summary.scalar('FW weights for option ' + str(MLPcounter), tf.math.reduce_mean(weight))

	def initializer_fn():
		if init_weights == 'xavier':
			return tf.contrib.layers.xavier_initializer(uniform = False)
		if init_weights == 'normal':
			return tf.random_normal_initializer()

	W = []
	b = []
	h = [x]


	with tf.name_scope('Weights'):

		#Initializing shared weights only for the first network, and reusing them for other networks
		if MLPcounter==1:
			shared_W = tf.get_variable('W' + str(1)+ '_' + str(MLPcounter), shape=[sdim, hidden_layer[0]], initializer=initializer_fn())
			shared_b = tf.get_variable('b' + str(1)+ '_' + str(MLPcounter), shape=[hidden_layer[0]], initializer=initializer_fn())
			shared_params = [shared_W, shared_b]
		else:
			shared_W = shared_params[0]
			shared_b = shared_params[1]
		W.append(shared_W)
		b.append(shared_b)
		h.append(activation_fn(tf.matmul(h[-1], W[0]) + b[0]))
		prev_layer_size = hidden_layer[0]

		
		for layer_number, layer_size in enumerate(hidden_layer[1:], start = 1):
			layer_number = layer_number+ 1 # Starting from 1 instead of 0
			# W.append(tf.Variable(tf.random_normal([prev_layer_size, layer_size]), name = 'W' + str(layer_number)+ '_' + str(MLPcounter)))
			# b.append(tf.Variable(tf.random_normal([layer_size]), name = 'b' + str(layer_number)+ '_' + str(MLPcounter)))
			# h.append(tf.nn.sigmoid(tf.matmul(h[-1], W[layer_number-1]) + b[layer_number-1]))

			W.append(tf.get_variable('W' + str(layer_number)+ '_' + str(MLPcounter), shape=[prev_layer_size, layer_size], initializer=initializer_fn()))
			b.append(tf.get_variable('b' + str(layer_number)+ '_' + str(MLPcounter), shape=[layer_size], initializer=initializer_fn()))
			h.append(activation_fn(tf.matmul(h[-1], W[layer_number-1]) + b[layer_number-1]))
			prev_layer_size = layer_size

		#h1 = tf.nn.dropout(h1, 0.5)
		tf.summary.scalar('Weights for option ' + str(MLPcounter), tf.math.reduce_mean(W[1]))

		# W_out = tf.Variable(tf.random_normal([hidden_layer[-1], adim]), name = 'Wout_' + str(MLPcounter))
		# b_out = tf.Variable(tf.random_normal([adim]), name = 'bout_' + str(MLPcounter))
		W_out = tf.get_variable('Wout_' + str(MLPcounter), shape=[hidden_layer[-1], adim], initializer=initializer_fn())
		b_out = tf.get_variable('bout_' + str(MLPcounter), shape=[adim], initializer=initializer_fn())
		
	mean = tf.matmul(h[-1], W_out) + b_out
	
	dist = tfd.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=cov_mat.astype(np.float32))
	dist2 = tfd.Normal(loc=mean, scale=cov_mat.astype(np.float32)[0,0])
	
	# Multiplying pdf with a small value compared to the observations for calculating probability
	# pdf = dist.prob(a)
	pdf2 = dist2.prob(a)
	# print('MVG ',pdf)
	# print('UVG ', pdf2)
	# y = tf.math.scalar_mul(0.1**k, pdf)
	y = tf.clip_by_value(pdf2, clip_value_min = 10**-37, clip_value_max = 1.0)
	y_prob = tf.math.reduce_prod(y, axis = 1)

	#Sknote: a1 log(pred) - a2 log(1-pred)
	print(y)
	logprob = tf.reduce_mean(tf.math.scalar_mul(-1, tf.reduce_sum(tf.math.log(y), axis=1)))
	tf.summary.scalar('Log probability ' + str(MLPcounter),logprob)
	
	# mean_loss = tf.math.reduce_mean(logprob)
	# tf.summary.scalar('Policy loss for option ' + str(MLPcounter), mean_loss)

	# wlogprob = tf.multiply(weight, logprob)
	wlogprob = tf.nn.l2_loss(mean- a)
		
	return {'state': x, 
				'action': a, 
				'weight': weight,
				'prob': y_prob, 
				'amax': mean,
				'lprob': logprob,
				'wlprob': wlogprob,
				'discrete': False}, shared_params



def conv2affine(sdim, adim, variance, _hiddenLayer=32):
	code = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))

	sarraydims = [s for s in sdim]
	sarraydims.insert(0, None)

	x = tf.placeholder(tf.float32, shape=sarraydims)

	a = tf.placeholder(tf.float32, shape=[None, adim])

	weight = tf.placeholder(tf.float32, shape=[None, 1])

	net = slim.conv2d(x, 64, [11, 11], 4, padding='VALID', scope='conv1'+code)
	#net = slim.conv2d(net, 192, [5, 5], scope='conv2'+code)
	#net = slim.conv2d(net, 384, [3, 3], scope='conv3'+code)

	net = slim.flatten(net)
	W1 = tf.Variable(tf.random_normal([68096, _hiddenLayer]))
	b1 = tf.Variable(tf.random_normal([_hiddenLayer]))
	output = tf.nn.sigmoid(tf.matmul(net, W1) + b1)
	#output= tf.nn.dropout(output, dropout)

	W2 = tf.Variable(tf.random_normal([_hiddenLayer, adim]))
	b2 = tf.Variable(tf.random_normal([adim]))

	output = tf.nn.sigmoid(tf.matmul(output, W2) + b2)

	logprob = tf.reduce_sum((output-a)**2, 1)/variance

	y = tf.exp(-logprob)

	wlogprob = tf.multiply(tf.transpose(weight), logprob)
		
	return {'state': x, 
				'action': a, 
				'weight': weight,
				'prob': y, 
				'amax':output,
				'lprob': logprob,
				'wlprob': wlogprob,
				'discrete': False}

def conv2mlp(sdim, adim, _hiddenLayer=32):
	code = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))

	sarraydims = [s for s in sdim]
	sarraydims.insert(0, None)

	x = tf.placeholder(tf.float32, shape=sarraydims)

	a = tf.placeholder(tf.float32, shape=[None, adim])

	weight = tf.placeholder(tf.float32, shape=[None, 1])

	net = slim.conv2d(x, 64, [11, 11], 4, padding='VALID', scope='conv1'+code)
	#net = slim.conv2d(net, 192, [5, 5], scope='conv2'+code)
	#net = slim.conv2d(net, 384, [3, 3], scope='conv3'+code)

	net = slim.flatten(net)
	W1 = tf.Variable(tf.random_normal([68096, _hiddenLayer]))
	b1 = tf.Variable(tf.random_normal([_hiddenLayer]))
	output = tf.nn.sigmoid(tf.matmul(net, W1) + b1)
	#output= tf.nn.dropout(output, dropout)

	W2 = tf.Variable(tf.random_normal([_hiddenLayer, adim]))
	b2 = tf.Variable(tf.random_normal([adim]))

	logit = tf.matmul(output, W2) + b2

	y = tf.nn.softmax(logit)

	logprob = tf.nn.softmax_cross_entropy_with_logits(logit, a)

	wlogprob = tf.multiply(tf.transpose(weight), logprob)
		
	return {'state': x, 
				'action': a, 
				'weight': weight,
				'prob': y, 
				'amax': tf.argmax(y, 1),
				'lprob': logprob,
				'wlprob': wlogprob,
				'discrete': True}


def conv2a3c(sdim, adim, _hiddenLayer=32):
	code = ''#.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))

	sarraydims = [s for s in sdim]
	sarraydims.insert(0, None)

	x = tf.placeholder(tf.float32, shape=sarraydims)

	a = tf.placeholder(tf.float32, shape=[None, adim])

	weight = tf.placeholder(tf.float32, shape=[None, 1])

	net = slim.conv2d(x, 16, [11, 11], 4, padding='VALID')
	net = slim.conv2d(x, 8, [3, 3], 2, padding='VALID')

	net = slim.flatten(net)
	W1 = tf.Variable(tf.random_normal([3200, _hiddenLayer]))
	b1 = tf.Variable(tf.random_normal([_hiddenLayer]))
	output = tf.nn.sigmoid(tf.matmul(net, W1) + b1)
	#output= tf.nn.dropout(output, dropout)

	W2 = tf.Variable(tf.random_normal([_hiddenLayer, adim]))
	b2 = tf.Variable(tf.random_normal([adim]))

	logit = tf.matmul(output, W2) + b2

	y = tf.nn.softmax(logit)

	logprob = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=a)

	wlogprob = tf.multiply(tf.transpose(weight), logprob)
		
	return {'state': x, 
				'action': a, 
				'weight': weight,
				'prob': y, 
				'amax': tf.argmax(y, 1),
				'lprob': logprob,
				'wlprob': wlogprob,
				'discrete': True}


def affine(sdim, adim, variance):
	"""
	This function creates a linear regression network that takes states and
	regresses to actions. It is based on a gated relu.

	Positional arguments:
	sdim -- int dimensionality of the state-space
	adim -- int dimensionality of the action-space
	variance -- float scaling for the probability calculation
	
	"""

	x = tf.placeholder(tf.float32, shape=[None, sdim])

	a = tf.placeholder(tf.float32, shape=[None, adim])

	weight = tf.placeholder(tf.float32, shape=[None, 1])

	W_h1 = tf.Variable(tf.random_normal([sdim, adim]))

	b_1 = tf.Variable(tf.random_normal([adim]))

	output = tf.matmul(x, W_h1) + b_1

	logprob = tf.reduce_sum((output-a)**2, 1)/variance

	y = tf.exp(-logprob)

	wlogprob = tf.multiply(tf.transpose(weight), logprob)
		
	return {'state': x, 
				'action': a, 
				'weight': weight,
				'prob': y, 
				'amax':output,
				'lprob': logprob,
				'wlprob': wlogprob,
				'discrete': False}


def gridWorldTabular(xcard, ycard, adim):
	"""
	This function creates a linear regression network that takes states and
	regresses to actions. It is based on a gated relu.

	Positional arguments:
	sdim -- int dimensionality of the state-space
	adim -- int dimensionality of the action-space
	variance -- float scaling for the probability calculation
	
	"""

	x = tf.placeholder(tf.float32, shape=[None, xcard, ycard])

	a = tf.placeholder(tf.float32, shape=[None, adim])

	weight = tf.placeholder(tf.float32, shape=[None, 1])

	table = tf.Variable(tf.abs(tf.random_normal([xcard, ycard, adim])))

	# Dimension: (None, xcard, ycard, adim)
	inputx = tf.tile(tf.reshape(x, [-1, xcard, ycard, 1]), [1, 1, 1, adim])

	# Dimension: (None, adim)
	collapse = tf.reduce_sum(tf.reduce_sum(tf.multiply(inputx, table), 1), 1)

	# Dimension: (None, 1)
	normalization = tf.reduce_sum(tf.abs(collapse),1)
	
	# Dimesnion: (None, adim), gives action probabilities
	actionsP = tf.abs(collapse) / tf.tile(tf.reshape(normalization, [-1, 1]), [1, adim])

	# Dimension: (None, 1)
	# I think it should be reduce sum
	y = tf.reduce_mean(tf.multiply(a, actionsP), 1)

	logprob = -tf.log1p(y)

	wlogprob = tf.multiply(tf.transpose(weight), logprob)
		
	return {'state': x, 
				'action': a, 
				'weight': weight,
				'prob': y, 
				'amax': tf.argmax(actionsP, 1),
				'debug': tf.multiply(a, actionsP),
				'lprob': logprob,
				'wlprob': wlogprob,
				'discrete': False}
	
