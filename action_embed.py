import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import affine
import h5py
import argparse
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import os

parser = argparse.ArgumentParser()
parser.add_argument('--read_file', help = "Format xxx_globalvvd.h5")
parser.add_argument('--save_dir', help = "Dir to save tensorboard plots")
parser.add_argument('--lr', help = "Learning rate")
parser.add_argument("--iterations")
parser.add_argument('--n_trajs')
parser.add_argument('--batch_size')
parser.add_argument('--embed_dim')
args = parser.parse_args()
learning_rate = 10**int(args.lr)
batch_size = int(args.batch_size)

EMBED_DIM = int(args.embed_dim)

TEMPERATURE = 1/100

# CENTER_SPINE_INDEX = 6
# BASE_TAIL_INDEX = 9
# BASE_NECK_INDEX = 3
NOSE = 0
LEFT_EAR = 1
RIGHT_EAR = 2
BASE_NECK_INDEX = 3
LEFT_FRONT_PAW = 4
RIGHT_FRONT_PAW = 5
CENTER_SPINE_INDEX = 6
LEFT_REAR_PAW = 7
RIGHT_REAR_PAW = 8
BASE_TAIL_INDEX = 9
MID_TAIL = 10
TIP_TAIL = 11	


# data = []
sas = []
# for filename in filelist[0]:
hf = h5py.File(args.read_file, 'r')
key_list = [key for key in hf.keys()]
for i in range(int(args.n_trajs)):
	print('Reading ' + str(i))
	# key = np.random.choice(key_list, 1)[0]
	key = key_list[i]
	pointtraj = hf.get(key+ '/points')[()]     # 240,7,2
	conftraj = hf.get(key+ '/confidence')[()]	
	velocitytraj = hf.get(key+ '/velocity')[()]	# 239,7,2

	for sas_tuple_number in range(len(velocitytraj)):
		s = np.reshape(pointtraj[sas_tuple_number], (-1))
		a = np.reshape(velocitytraj[sas_tuple_number], (-1))
		s_dash = np.reshape(pointtraj[sas_tuple_number+1], (-1))
		sas.append((s, a, s_dash))
	# traj = [pointtraj, conftraj, velocitytraj]
	# data.append(traj)
sas = np.array(sas)

def sample_batch(batch_size):

	indices = np.random.choice(np.arange(len(sas)), batch_size, replace = False)
	sas_tuples = sas[indices]
	s = sas_tuples[:, 0]
	a = sas_tuples[:, 1]
	s_dash = sas_tuples[:, 2]
	feed_dict = {"x": np.concatenate([s, s_dash]), "a": np.array(a)}
	# feed_dict["x"] = np.concatenate([s, s_dash])
	# feed_dict["a"] = a
	return feed_dict


def cosine_similarity(t1, t2):
	normalize_t1 = tf.nn.l2_normalize(t1,0)        
	normalize_t2 = tf.nn.l2_normalize(t2,0)
	cos_similarity = tf.reduce_sum(tf.multiply(normalize_t1,normalize_t2))
	return cos_similarity

def _random_choice(inputs, n_samples):
    """
    With replacement.
    Params:
      inputs (Tensor): Shape [n_states, n_features]
      n_samples (int): The number of random samples to take.
    Returns:
      sampled_inputs (Tensor): Shape [n_samples, n_features]
    """
    # (1, n_states) since multinomial requires 2D logits.
    uniform_log_prob = tf.expand_dims(tf.zeros(tf.shape(inputs)[0]), 0)

    ind = tf.multinomial(uniform_log_prob, n_samples)
    ind = tf.squeeze(ind, 0, name="random_choice_ind")  # (n_samples,)

    return tf.gather(inputs, ind, name="random_choice")


def embed(sdim, adim, embed_dim, batch_size):

	sess = tf.Session()

	with tf.name_scope('embed_network'):
		x = tf.placeholder(tf.float32, shape=[None, 2*sdim])

		# This network omputes the embedding
		W_embed = tf.get_variable("W_embed", shape=[2*sdim, embed_dim], 
									initializer = tf.random_normal_initializer(stddev = 5))
		b_embed = tf.get_variable("b_embed", shape=[embed_dim], 
									initializer = tf.random_normal_initializer(stddev = 5))
		embedding = tf.math.add(tf.matmul(x, W_embed), b_embed, name = "embedding")
		tf.summary.scalar('embedding', tf.reduce_mean(embedding))

		# Computing how similar the embedding is to the action [g(Et =e|St, St+1) ]
		norm_ = tf.norm(embedding, axis= 1, keepdims = True)
		tf.summary.scalar('norm', tf.reduce_min(tf.math.abs(norm_)))
		embed_norm = tf.math.divide(embedding, tf.tile(norm_, tf.constant([1, embed_dim], tf.int32)), name = "embed_norm")
		tf.summary.scalar('embed_norm', tf.reduce_mean(embed_norm))
		cosine_matrix = -(tf.matmul(embed_norm, tf.transpose(embed_norm), name = "cosine_matrix"))
		tf.summary.scalar('cosine_matrix', tf.reduce_mean(cosine_matrix))
		exp_cosine_matrix = tf.math.exp(cosine_matrix, name = "exp_cosine_matrix")
		tf.summary.scalar('exp_cosine_matrix', tf.reduce_mean(exp_cosine_matrix))
		#z_vector should be a vector of ones for this case
		z_vector = tf.math.exp(tf.math.reduce_sum(tf.math.multiply(embed_norm, embed_norm), axis = 1), 
								name = "z_vector")
		normalising_vector = tf.math.reduce_sum(exp_cosine_matrix, axis = 1, name = "normalising_vector")
		tf.summary.scalar('normalising_vector', tf.reduce_mean(normalising_vector))
		g_hat_vector = tf.math.divide(z_vector, normalising_vector)

		# Computing f(At|Et =e)
		a = tf.placeholder(tf.float32, shape=[None, adim])
		W_f = tf.get_variable("W_f", shape=[embed_dim, adim], initializer = tf.random_normal_initializer())
		b_f = tf.get_variable("b_f", shape=[adim], initializer = tf.random_normal_initializer())
		z_f = tf.matmul(embedding, W_f) + b_f
		a_norm = tf.math.divide(a, tf.tile(tf.norm(a, axis= 1, keepdims = True), tf.constant([1, adim], tf.int32)))
		tf.summary.scalar("a_norm", tf.reduce_mean(a_norm))
		z_f_norm = tf.math.divide(z_f, tf.tile(tf.norm(z_f, axis= 1, keepdims = True), tf.constant([1, adim], tf.int32)))
		cosine_matrix2 = -(tf.matmul(z_f_norm, tf.transpose(a_norm), name = "cosine_matrix2"))
		exp_cosine_matrix2 = tf.math.exp(cosine_matrix2)
		z_f_vector = tf.math.exp(tf.math.reduce_sum(tf.math.multiply(a_norm, z_f_norm), axis =1), name = "z_f_vector")
		normalising_vector2 = tf.math.reduce_sum(exp_cosine_matrix2, axis = 1, name = "normalising_vector2")
		f_hat_vector = tf.math.divide(z_f_vector, normalising_vector2)

		# Pˆ(At|St, St+1)
		p_hat = tf.multiply(f_hat_vector, g_hat_vector)
		tf.summary.scalar('Probability', tf.reduce_mean(p_hat))
		p_hat_clipped = tf.clip_by_value(p_hat, clip_value_min = 10**-37, clip_value_max = 1.0, name = "p_hat_clipped")

		loss0 = tf.math.reduce_mean(-tf.math.log(p_hat_clipped))
		loss = tf.reduce_mean(loss0, name = "loss")

		# prob = tf.clip_by_value(prob0, clip_value_min = 10**-37, clip_value_max = 1.0)
		
	# loss = tf.reduce_mean(-tf.math.log(prob))
	optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
	train = optimizer.minimize(loss)

	sess.run(tf.global_variables_initializer())

	tf.summary.scalar('Loss', loss)
	writer = tf.summary.FileWriter(os.path.join(args.save_dir, "tb"))
	writer.add_graph(sess.graph)

	for it in range(int(args.iterations)):

		print("Iteration: ", it)

		# batch = sample_batch(128)
		indices = np.random.choice(np.arange(len(sas)), batch_size, replace = False)
		sas_tuples = sas[indices]
		s = sas_tuples[:, 0]
		a_ = sas_tuples[:, 1]
		s_dash = sas_tuples[:, 2]
		states = np.concatenate([s, s_dash], axis = 1)
		# print("states ", states.shape)
		feed_dict = {x: states, a: a_}

		sess.run(train, feed_dict)
		print("Loss: ", sess.run(loss, feed_dict))
		merged = tf.summary.merge_all()
		s = sess.run(merged, feed_dict)
		writer.add_summary(s, it)


embed(14, 14, EMBED_DIM, batch_size)
