from gabe_final.neural_net import *
import copy

def test_initialize_W():
	W,b,a = initialize_W(8,3,8)
	assert(len(W) == 2)
	assert(len(a) == 3)
	assert(len(b) == 2)
	
def test_f():
	test_array = np.random.random((5,5))
	test_out = f(test_array)

	for i in range(5):
		for j in range(5):
			assert(test_out[i,j] == sigmoid(test_array[i,j]))

