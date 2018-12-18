import numpy as np
import pdb
import collections
import datetime
import os
import json
import itertools
from keras import metrics
from keras.callbacks import EarlyStopping 
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import LSTM, Dense, Dropout, Concatenate, Input, Reshape
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

LSTM_DIM_SIZE = 32
NUM_CLASSES = 20
BASELINE_SCORE = (1.0 / NUM_CLASSES) 
NUM_FEATURES = 2


MAX_DESCENDING = 703.0 
MAX_ASCENDING = 144.0
# if you run find_max.py you'll see that the maximum packet size for ascending packets is 144
# and for descending packets is 703. We're going to use that for doing feature scaling



def create_model_multi(MAX_SEQ_LEN):
	
	model = Sequential()
	model.add(LSTM(LSTM_DIM_SIZE, input_shape=(MAX_SEQ_LEN, NUM_FEATURES)))
	model.add(Dropout(0.2))
	model.add(Dense(units=NUM_CLASSES, activation='softmax'))

	model.compile(loss='categorical_crossentropy',
		optimizer='adam',
		metrics=['accuracy'])

	return model


def create_model_single(MAX_SEQ_LEN, hyperparameter):
	
	model = Sequential()
	model.add(Reshape((1, MAX_SEQ_LEN), input_shape=(MAX_SEQ_LEN,1)))

	for i in range(hyperparameter.nb_layers):
		model.add(LSTM(
			units=LSTM_DIM_SIZE,
			activation=hyperparameter.activation_function,
			dropout=hyperparameter.dropout,
			return_sequences=True
		))
	
	model.add(LSTM(
		units=LSTM_DIM_SIZE,
		activation=hyperparameter.activation_function,
		dropout=hyperparameter.dropout
	))

	model.add(Dense(NUM_CLASSES, activation="softmax"))

	optimizer = hyperparameter.optimizer_builder(
		lr= hyperparameter.lr, 
		decay = hyperparameter.decay 
	)


	model.compile(loss='sparse_categorical_crossentropy',
		optimizer=optimizer,
		metrics=[metrics.sparse_categorical_accuracy])#,
		#class_mode = 'binary')

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
	#https://en.wikipedia.org/wiki/Feature_scaling
	def scale_feature(x):
		return (np.float64(x) + MAX_DESCENDING) / (MAX_DESCENDING + MAX_ASCENDING)

	def scale_feature_divide_by_each_max(x):
		x = np.float64(x)
		return x / MAX_ASCENDING if x > 0 else x / MAX_DESCENDING

	def alternate_received_and_sent():
		newlist = []
		def treat_element(x):
			newlist.append(scale_feature_divide_by_each_max(x))

		for item in olist[::-1]:
			if item == 1:
				treat_element(slist.pop())
			else:
				treat_element(rlist.pop() * -1)
		return newlist[::-1]

	def concat():
		return np.concatenate((np.array(slist_param)*-1, np.array(rlist_param)))
		#TODO in the paper say we tried concatenating the sent and the received it gives 10% less accuracy

	return alternate_received_and_sent()



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
	features = pad_sequences(features, dtype="float64", maxlen=MAX_SEQ_LEN)
	return np.array(features), np.array(labels), MAX_SEQ_LEN


DataInfo = collections.namedtuple("DataInfo", "features labels max_len")

class EarlyStoppingOnBatch(EarlyStopping):
	def on_epoch_end(self, epoch, logs=None):
		pass
	def on_batch_end(self, batch, logs=None):
		current = self.get_monitor_value(logs)
		if current is None:
			return

		if self.monitor_op(current - self.min_delta, self.best):
			self.best = current
			self.wait = 0
			if self.restore_best_weights:
				self.best_weights = self.model.get_weights()
		else:
			self.wait += 1
			if self.wait >= self.patience:
				self.batch = batch 
				self.model.stop_training = True
			if self.restore_best_weights:
				if self.verbose > 0:
					print('Restoring model weights from the end of '
						'the best batch')
				self.model.set_weights(self.best_weights)

def find_mean_and_standard_derivation(xs):
	as_single_list = np.reshape(xs, (-1))	
	mean = np.mean(as_single_list)
	standard_derivation = np.std(as_single_list)
	return mean, standard_derivation

def standardize(xs, mean, standard_derivation):	
	return (xs - mean) / standard_derivation


def single_feature(dataInfo, hyperparameter):
	features, olabels, max_len = dataInfo

	features = np.reshape(features, [features.shape[0], max_len, 1]) #Add a dimension so keras is happy
	X_train, X_test, y_train, y_test = train_test_split(features, olabels, test_size=.2)#shuffles the data by default
	y_train = np.reshape(y_train, [len(y_train), 1])

	model = create_model_single(max_len, hyperparameter)
	print(model.summary())
	early_stopping = EarlyStoppingOnBatch(monitor='loss' , min_delta=0.001, patience=25, verbose=0, mode='auto', baseline=0.01, restore_best_weights=False)
	fit_return = model.fit(X_train, y_train, batch_size=hyperparameter.batch_size, epochs=hyperparameter.epochs, callbacks=[early_stopping], validation_split= 0.15, shuffle= 'batch')

	score = model.evaluate(X_test, y_test)
	y_pred = model.predict_classes(X_test)
	ilabels = y_pred
	print("correct labels were", y_test, "infered labels are", ilabels)
	res = accuracy_score(y_test, y_pred)
	print("accuracy is", res)
	return res

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
	print(len(X_train), len(y_train), len(X_test), len(y_test))
	model = create_model_multi(max_len)
	model.fit(X_train, y_train, batch_size=16, epochs=1)
	score = model.evaluate(X_test, y_test)
	print(score)
	y_pred = one_in_max_of_cols(model.predict(X_test))
	print(accuracy_score(y_test, y_pred))

def create_sequence(min_val, max_val, number_steps):
	i = 0
	sequence = []
	step_size = (max_val - min_val) / number_steps
	while i < number_steps:
		sequence.append(min_val + step_size * i)
		i+=1 
	return sequence

#truncation index is the length at which we discard the features inputs
Hyperparameter = collections.namedtuple("Hyperparameter", "nb_layers decay optimizer_builder lr batch_size epochs dropout activation_function")

def create_possible_hyperparameters():
	number_steps = 3

	decays = create_sequence(0, 0.9, number_steps)


	optimizer_builders = [SGD, Adam, RMSprop]
	learning_rates = create_sequence(0.0001, 0.1, number_steps)

	batch_sizes = create_sequence(32, 256, number_steps)	
	possible_epochs = create_sequence(1, 10, number_steps)
	possible_nb_layers = [0,1,2] #TODO add it after
	dropouts = create_sequence(0.1, 0.5, number_steps)
	
	activation_functions = ["sigmoid", "relu", "tanh"] 

	
	cartesian_prod_result = itertools.product(possible_nb_layers, decays, optimizer_builders, learning_rates, batch_sizes, 
		possible_epochs, #possible_nb_layers, TODO add this param
		dropouts, activation_functions)
	hyperparameters = []
	for hyperparameter_tuple in cartesian_prod_result:
		hyperparameters.append(Hyperparameter(*hyperparameter_tuple))

	print("There are {0} possible hyperparameters\n\n".format(len(hyperparameters)))
	return hyperparameters
	
	
if __name__ == '__main__':

	# urls = []
	# url_file = "short_list_500"
	# with open(url_file) as f:
	# 	lines = f.readlines()
	# 	for line in lines:
	# 		urls.append(line.strip())

	datadir = "../data_cw"+str(NUM_CLASSES)+"_day0_to_30/"
	hyperparameters = create_possible_hyperparameters()
	np.random.shuffle(hyperparameters)
	hyperparameter_to_score = {}
	dataInfo = DataInfo(*get_data_single(datadir))

	def get_dated_file_name(prefix):
		now = datetime.datetime.utcnow()
		return "{0}_d{1}_{2}h_{3}m".format(prefix, now.day, now.hour, now.minute)

	log_file_name = get_dated_file_name("../logs/log_train")


	result_filename = get_dated_file_name("../results/result_file")
	def update_result_file(nb_tried):
		print("UPDATING RESULT FILE")
		sorted_keys = sorted(hyperparameter_to_score, reverse=True)
		with open(result_filename, "w") as f:
			f.write("tried the first {0} hyperparameters\n".format(nb_tried))
			f.write("accuracy_score\thyperparameter\n")
			for key in sorted_keys: 
				f.write("{0}\t{1}\n".format(key, hyperparameter_to_score.get(key)))

	i = 0
	with open(log_file_name, "a") as log_file:
		for hyperparameter in hyperparameters:
			print("\nusing this hyperparameter: "+ str(hyperparameter)+"\n")

			accuracy_score_value = single_feature(dataInfo, hyperparameter)
			if accuracy_score_value == None:
				print("=============DISCARDING MODEL============")
				continue

			result = {accuracy_score_value: hyperparameter}
			log_file.write("accuracy score: {0}, hyperparameter: {1}\n".format(accuracy_score_value, hyperparameter))
			hyperparameter_to_score.update(result)
			update_result_file(i+1)

			i+=1
