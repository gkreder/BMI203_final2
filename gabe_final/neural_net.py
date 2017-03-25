import numpy as np
import copy
from .io import *
import random
import pickle as pkl
import csv

EPSILON = 0.01

def sigmoid(z):
	"""
	sigmoid activation function on single number
	"""
	ans = 1.0 / (1.0 + np.exp(-1.0 * z))
	return ans

def f(z):
	"""
	vector form of sigmoid activation function
	"""
	return np.vectorize(sigmoid)(z)

def f_prime(z):
	"""
	Vector form of f_prime
	"""
	f_prime = f(z)
	# print(1.0 - f_prime)
	# f_prime = np.multiply(f_prime, (1 - f_prime))
	f_prime = f_prime * (1 - f_prime)
	# print(f_prime)
	return(f_prime)

def initialize_W(input_layer_size, hidden_layer_size, output_layer_size):
	"""
	Returns empty neural net (Weights (W), biases (b), and activations (a))
	"""
	input_hidden_weights = np.random.normal(loc = 0.0, scale = EPSILON ,size = (hidden_layer_size, input_layer_size))
	input_hidden_biases = np.random.normal(loc = 0.0, scale = EPSILON ,size = (hidden_layer_size,1))

	hidden_output_weights = np.random.normal(loc = 0.0, scale = EPSILON ,size = (output_layer_size, hidden_layer_size))
	hidden_output_biases = np.random.normal(loc = 0.0, scale = EPSILON ,size = (output_layer_size,1))

	input_activations = np.zeros((input_layer_size, 1))
	hidden_activations = np.zeros((hidden_layer_size, 1))
	output_activations = np.zeros((output_layer_size, 1))

	W = [input_hidden_weights, hidden_output_weights]
	b = [input_hidden_biases, hidden_output_biases]
	a = [input_activations, hidden_activations, output_activations]

	return W,b,a

def initialize_delta_W(W, b):
	"""
	Initialize delta_W and delta_b to be empty and same sizes/shapes
	as W and b
	"""
	delta_W = []
	delta_b = []
	for col in W:
		delta_W.append(np.zeros(np.shape(col)))
	for col in b:
		delta_b.append(np.zeros(np.shape(col)))
	return delta_W, delta_b

def forward_propagate(W,b,a):
	"""
	Given W,b, and a, run forward propagation
	"""
	for l in range(0, len(a) - 1):
		W_l = W[l]
		a_l = a[l]
		b_l = b[l]

		z_next = np.dot(W_l, a_l) + b[l]
		a_next = f(z_next)
		a[l + 1] = a_next

	return W,b,a

def back_propagate(W, b, a, x, y):
	"""
	Given current state of neural net (W,b,a) and input/output data (x and y)
	run back propagation 
	"""
	del_W = []
	del_b = []
	deltas = []


	# a[0] = bits_to_vector(x)
	a[0] = x
	W, b, a = forward_propagate(W,b,a)
	# vec_y = bits_to_vector(y)
	vec_y = y
	nl = len(a)
	delta_nl = (vec_y - a[nl - 1]) * -1.0
	z = a[nl - 1]
	f_prime_nl = f_prime(z)
	delta_nl = np.multiply(delta_nl, f_prime_nl)
	deltas.append(delta_nl)

	for l in range(len(W) - 1, 0, -1):
		W_T = np.transpose(W[l])
		delta = np.dot(W_T, deltas[-1])
		f_prime_current = f_prime(a[l])
		delta = np.multiply(delta, f_prime_current)
		deltas.append(delta)

	deltas = deltas[::-1]
	for l in range(0, len(W)):
		a_T = np.transpose(a[l])
		del_current = deltas[l]
		del_W.append(np.dot(del_current, a_T))
		del_b.append(del_current)

	return del_W, del_b, W, b, a
		
def batch_gradient_descent(input_layer_size, hidden_layer_size, output_layer_size, training_data, alpha, lamb, number_iterations = 1000):
	"""
	Run batch_gradient descent to train auto-encoder with given layer sizes. Run for NUM_ITERATIONS number
	of iterations. Code adapted from equations in
	http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/
	"""
	W,b,a = initialize_W(input_layer_size, hidden_layer_size, output_layer_size)
	delta_W, delta_b = initialize_delta_W(W, b)
	m = len(training_data)
	n_l = len(a) - 1
	m_recip = 1 / float(m)

	NUM_ITERATIONS = number_iterations
	for j in range(NUM_ITERATIONS):
		print('Iteration: ' + str(j + 1) + ' of ' + str(NUM_ITERATIONS))
		# W,b,a = round_all(W,b,a)
		delta_W, delta_b = initialize_delta_W(W, b)
		
		for i in range(0, m):
			x, y = training_data[i]
			del_W, del_b, W, b, a = back_propagate(W,b,a,x,y)
			for l in range(n_l):
				delta_W[l] = delta_W[l] + del_W[l]
				delta_b[l] = delta_b[l] + del_b[l]


		for l in range(n_l):
			# W_alpha_term = m_recip * delta_W[l]
			# W_lambda_term = lamb * W[l]
			# W_sub = alpha * (W_alpha_term + W_lambda_term)
			# b_alpha_term = m_recip * delta_b[l]
			# b_sub = alpha * b_alpha_term
			# W[l] = W[l] - W_sub
			# b[l] = b[l] - b_sub

			W[l] = W[l] - delta_W[l]
			b[l] = b[l] - delta_b[l]

	return W,b,a


def train_autoencoder(input_layer_size, hidden_layer_size, output_layer_size):
	"""
	Trains an autoencoder with specified input layer size, hidden layer size, and output
	layer size to take an 8 digit number with all 0's except for a single 1
	"""
	alpha = 0.01
	lamb = 0.1
	test_data = read_test_data()
	test_data_vecs = training_data_to_vecs(test_data)
	W,b,a = batch_gradient_descent(input_layer_size, hidden_layer_size, output_layer_size, test_data_vecs, alpha, lamb)
	print(test_data_vecs)
	return W,b,a


def train_DNA_net(hidden_layer_size, number_iterations, training_data):
	"""
	Uses training data of DNA sequences (length 17) of positive  and negative Rap1 
	binding sites to train a 68x20x1 neural net. Encodes sequences as 68-bit
	vectors by mapping A to 1000, G to 0100, T to 0010, and C to 0001. 
	"""
	alpha = 0.01
	lamb = 0.1
	W,b,a = batch_gradient_descent(68, hidden_layer_size, 1, training_data, alpha, lamb, number_iterations = number_iterations)
	pkl.dump((W,b,a), open('DNA_net.pkl', 'wb'))
	return W,b,a

def likelihood(DNA_seq, trained_W, trained_b, a, vec = False):
	"""
	Calculate likelihood score of a DNA sequence given a trained
	neural network with weights trained_W and biases trained_b. 
	Likelihood is calculated as the difference between the input
	vector and output vector. Since the network has been trained on
	positive vectors, input vectors that are positive are more likely
	to produce output vectors that are closer to the original vector
	"""
	if not vec:
		DNA_vec = DNA_to_vector(DNA_seq)
	else:
		DNA_vec = DNA_seq
	a[0] = DNA_vec
	output = forward_propagate(trained_W,trained_b,a)[2][2]
	return output

def test_predictions(pickle = True):
	"""
	For every sequence in file rap1-lieb-test.txt, calculate likelihood score
	using likelihood function above and print results
	"""
	sequences = read_test_DNA()
	predictions = []
	training_data = create_training_data()
	if pickle:
		W,b,a = pkl.load(open('DNA_encoder.pkl', 'rb'))
	else:
		W,b,a = train_DNA_net(10, 1000, training_data)

	output_string = ''
	with open('likelihood_scores.tsv', 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter = '\t')
		for sequence in sequences:
			output_string += sequence + '\t' + str(likelihood(sequence, W,b,a)[0][0]) + '\t'
			# print(sequence, '\t', likelihood(sequence, W,b,a))
			writer.writerow([sequence, str(likelihood(sequence, W,b,a)[0][0])])
	print(output_string)


def cross_validation():
	"""
	Perform cross-valdation by breaking available data and picking a random subset of
	the negative sequences such that the ratio of negative to positive sequences in 
	the data is 10:1. Take this data and divide into 5 subsections. Use each of the 5
	subsections and use it as the training set, then evaluate the resulting 
	neural net's performance
	"""
	training_data = create_training_data()
	random.shuffle(training_data)
	data_1 = training_data[0 : 300]
	data_2 = training_data[300 : 600]
	data_3 = training_data[600 : 900]
	data_4 = training_data[900 : 1200]
	data_5 = training_data[1200 : ]

	data_sets = [data_1, data_2, data_3, data_4, data_5]

	hidden_layer_size = 10
	number_iterations = 100

	for i in range(5):
		training_set = data_sets[i]
		if i == 0:
			test_set = data_2 + data_3 + data_4 + data_5
		elif i == 1:
			test_set = data_1 + data_3 + data_4 + data_5
		elif i == 2:
			test_set = data_1 + data_2 + data_4 + data_5
		elif i == 3:
			test_set = data_1 + data_2 + data_3 + data_5
		elif i == 4:
			test_set = data_1 + data_2 + data_3 + data_4

		trained_W,trained_b,a = train_DNA_net(hidden_layer_size, number_iterations, training_data)

		true_num_positives = 0
		guess_num_positives = 0
		true_num_negatives = 0
		guess_num_negatives = 0

		for seq, positive in test_set:
			guess = likelihood(seq, trained_W, trained_b, a, vec = True)[0][0]
			# if hit:
			if positive:
				true_num_positives += 1
			else:
				true_num_negatives += 1
			if guess > 0.5:
				guess_num_positives += 1
			else:
				guess_num_negatives += 1

		results = (true_num_positives, true_num_negatives, guess_num_positives, guess_num_negatives)

		f_name = 'results_training_' + str(i) + '.pkl'
		pkl.dump(results, open(f_name, 'wb'))

def vary_hidden_layer_size():
	"""
	Test effect of varying hidden layer size on network behavior. 
	Holds number of iterations constant at 100 and trains neural
	nets with varying hidden layer size. The response to 
	training data is then calculated and a basic response
	metric is saved
	"""
	training_data = create_training_data()
	random.shuffle(training_data)
	data_1 = training_data[0 : 300]
	data_2 = training_data[300 : 600]
	data_3 = training_data[600 : 900]
	data_4 = training_data[900 : 1200]
	data_5 = training_data[1200 : ]
	data_sets = [data_1, data_2, data_3, data_4, data_5]
	training_set = data_sets[0]
	test_set = data_2 + data_3 + data_4 + data_5
	layer_sizes = [1,5,10,20,50,100]
	number_iterations = 100

	for hidden_layer_size in layer_sizes:
		trained_W,trained_b,a = train_DNA_net(hidden_layer_size, number_iterations, training_data)
		true_num_positives = 0
		guess_num_positives = 0
		true_num_negatives = 0
		guess_num_negatives = 0

		for seq, positive in test_set:
				guess = likelihood(seq, trained_W, trained_b, a, vec = True)[0][0]
				# if hit:
				if positive:
					true_num_positives += 1
				else:
					true_num_negatives += 1
				if guess > 0.5:
					guess_num_positives += 1
				else:
					guess_num_negatives += 1

		results = (true_num_positives, true_num_negatives, guess_num_positives, guess_num_negatives)
		f_name = 'results_layer_size_' + str(hidden_layer_size) + '.pkl'
		pkl.dump(results, open(f_name, 'wb'))

def vary_iterations():
	"""
	Test effect of varying iteration number on network behavior. 
	Holds hidden layer size constant at 10 and trains neural
	nets with varying number of iterations. The response to 
	training data is then calculated and a basic response
	metric is saved
	"""
	training_data = create_training_data()
	random.shuffle(training_data)
	data_1 = training_data[0 : 300]
	data_2 = training_data[300 : 600]
	data_3 = training_data[600 : 900]
	data_4 = training_data[900 : 1200]
	data_5 = training_data[1200 : ]
	data_sets = [data_1, data_2, data_3, data_4, data_5]
	training_set = data_sets[0]
	test_set = data_2 + data_3 + data_4 + data_5
	number_iterations_list = [1,10,50,100,1000] 
	hidden_layer_size = 10

	for number_iterations in number_iterations_list:
		trained_W,trained_b,a = train_DNA_net(hidden_layer_size, number_iterations, training_data)
		true_num_positives = 0
		guess_num_positives = 0
		true_num_negatives = 0
		guess_num_negatives = 0

		for seq, positive in test_set:
				guess = likelihood(seq, trained_W, trained_b, a, vec = True)[0][0]
				# if hit:
				if positive:
					true_num_positives += 1
				else:
					true_num_negatives += 1
				if guess > 0.5:
					guess_num_positives += 1
				else:
					guess_num_negatives += 1

		results = (true_num_positives, true_num_negatives, guess_num_positives, guess_num_negatives)
		f_name = 'results_iterations_' + str(number_iterations) + '.pkl'
		pkl.dump(results, open(f_name, 'wb'))


