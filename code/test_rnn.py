from classify_rnn import *
import numpy as np
def test_one_in_max_of_cols():
	x = np.array([
		[20, 3,  4 ],
		[4,  30, 40],
		[2,  4,  10],
		[1,  15, 5 ]])
	expected = np.array([
		[1,0,0],
		[0,1,1],
		[0,0,0],
		[0,0,0]])

	assert np.array_equal(expected, one_in_max_of_cols(x))
		
