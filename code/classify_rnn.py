import numpy as np
import os
import json
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Concatenate, Input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

BATCH_SIZE = 16
EPOCHS = 100

LSTM_DIM_SIZE = 32
NUM_CLASSES = 1500
NUM_FEATURES = 2


def create_model_multi(MAX_SEQ_LEN):
	
	model = Sequential()
	model.add(LSTM(LSTM_DIM_SIZE, input_shape=(MAX_SEQ_LEN, NUM_FEATURES)))
	model.add(Dropout(0.2))
	model.add(Dense(units=NUM_CLASSES, activation='softmax'))

	model.compile(loss='categorical_crossentropy',
		optimizer='adam',
		metrics=['accuracy'])

	return model


def create_model_single(MAX_SEQ_LEN):
	
	model = Sequential()
	model.add(LSTM(LSTM_DIM_SIZE, input_shape=(MAX_SEQ_LEN, 1)))
	model.add(Dropout(0.2))
	model.add(Dense(NUM_CLASSES, activation='sigmoid'))

	model.compile(loss='categorical_crossentropy',
		optimizer='adam',
		metrics=['accuracy'])

	return model


#zips the sent and received data of pcaps structs, outputs a list of this pcaps traces (list) through all the files.
# -> aggregates data from different files
#saves an array of pcap identifiers in-order with the corresponding pcap list
# list of list( (s1 r1 s2 r2 .....) )
def get_data_multi(data_folder, number_files_taken=None):
	
	flist = os.listdir(data_folder)
	if number_files_taken != None:
		flist = flist[0 : min(len(flist), number_files_taken)]
	features = []
	labels = []
	MAX_SEQ_LEN = 0
	for fname in flist:
		print(fname)
		with open(data_folder + fname) as f:
			data_dict = json.loads(f.read())
			for k, v in sorted(data_dict.items()):
				# TODO(sandra): Offset and normalize times to [0, 1]
				feature = zip(v['sent'], v['received']) #Jules : does this discard the last elements of the longest list ?
				if len(v['received']) > MAX_SEQ_LEN:
					MAX_SEQ_LEN = len(v['received'])
				features.append(list(feature)) #Putting all features
				labels.append(int(k[:-5]))
	features = pad_sequences(features, maxlen=MAX_SEQ_LEN)
	return np.array(features), np.array(labels), MAX_SEQ_LEN

def make_single_feature(slist, rlist, olist): 
	#reconstructs the traffic from the received, sent, order lists

	newlist = []
	for item in olist:
		if item == 1:
			newlist.append(slist.pop())
		else:
			newlist.append(rlist.pop() * -1)
	return newlist

def get_data_single(data_folder):
	#list of list( (s1 -r1 -r2 s2 -r3 s3 s4 ...) )
	
	flist = os.listdir(data_folder)
	features = []
	labels = []
	MAX_SEQ_LEN = 0
	for fname in flist:
		print(fname)
		with open(data_folder + fname) as f:
			data_dict = json.loads(f.read())
			for k, v in data_dict.items():
				new_list = make_single_feature(v['sent'], v['received'], v['order'])
				# rec_list = v['received']
				# if len(rec_list) > MAX_SEQ_LEN:
				# 	MAX_SEQ_LEN = len(rec_list)
				#features.append(rec_list) #Currently trying only with received lengths
				if len(new_list) > MAX_SEQ_LEN:
					MAX_SEQ_LEN = len(new_list)
				features.append(new_list)
				labels.append(int(k[:-5]))
	features = pad_sequences(features, maxlen=MAX_SEQ_LEN)
	return np.array(features), np.array(labels), MAX_SEQ_LEN


def single_feature(data_folder):

	features, labels, max_len = get_data_single(data_folder)
	#features[:, :, 0] /= np.max(features[:, :, 0])
	#features[:, :, 1] /= np.max(features[:, :, 1])
	features = np.reshape(features, [features.shape[0], max_len, 1])
	labels = convert_labels(labels)
	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.2)
	print len(X_train), len(y_train), len(X_test), len(y_test)
	model = create_model_single(max_len)
	model.fit(X_train, y_train, batch_size=16, epochs=1)
	score = model.evaluate(X_test, y_test)
	print(score)
	y_pred = model.predict(X_test)
	print accuracy_score(y_test, y_pred)

def convert_labels(Y):
	#one-hot
	# encoder = LabelEncoder()
	# encoder.fit(Y)
	# encoded_Y = encoder.transform(Y)
	# new_y = np_utils.to_categorical(encoded_Y)
	new_y = to_categorical(Y, num_classes=NUM_CLASSES)

	return new_y

#this function takes a matrix
#of dimension N*D and transforms each column
# such that only a one is left in each column.
# the one is left in the column where the maximum value of that column is located.
# e.g. if input is 
#     [[.4 .5 .7]
#      [.2 .9 .3]
#      [.1 .8 .3]]
#output would be 
#     [[1  0  1]
#      [0  1  0]
#      [0  0  0]]	
def one_in_max_of_cols(matrix):
	row_index_of_maximums = np.argmax(matrix, axis=0)
	first_column = matrix[:,0]
	new_columns = []
	for max_index in row_index_of_maximums: 
	    zeros = np.zeros_like(first_column)
	    zeros[max_index] = 1
	    new_columns.append(zeros)
	return np.stack(new_columns, axis=1)

def multi_feature(data_folder, number_files_for_training=None):

	features, labels, max_len = get_data_multi(data_folder, number_files_for_training)
	#To-do
	#features[:, :, 0] /= np.max(features[:, :, 0])
	#features[:, :, 1] /= np.max(features[:, :, 1])
	features = np.reshape(features, [features.shape[0], max_len, NUM_FEATURES])
	labels = convert_labels(labels)
	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.2)
	print len(X_train), len(y_train), len(X_test), len(y_test)
	model = create_model_multi(max_len)
	model.fit(X_train, y_train, batch_size=16, epochs=1)
	score = model.evaluate(X_test, y_test)
	print(score)
	y_pred = one_in_max_of_cols(model.predict(X_test))
	print accuracy_score(y_test, y_pred)
	
if __name__ == '__main__':

	# urls = []
	# url_file = "short_list_500"
	# with open(url_file) as f:
	# 	lines = f.readlines()
	# 	for line in lines:
	# 		urls.append(line.strip())

	#single_feature()
	datadir = "../data_cw100_day0_to_10/"
	multi_feature(datadir)

