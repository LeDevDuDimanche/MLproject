import numpy as np
import inspect 
import collections
from guppy import hpy;
import datetime
import os
import json
from hyperopt import fmin, tpe, hp, Trials
import itertools
from create_closed_world import closed_world_foldername, create_closed_world
from utils.util import get_bursts, ngrams_bursts
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

NUM_FEATURES = 2
BURST = False


MAX_DESCENDING = 703.0 
MAX_ASCENDING = 144.0
# if you run find_max.py you'll see that the maximum packet size for ascending packets is 144
# and for descending packets is 703. We're going to use that for doing feature scaling

#truncation index is the length at which we discard the features inputs
class Hyperparameter:
	def __init__(self, nb_layers, decay, optimizer_builder, lr, batch_size, epochs, dropout, activation_function, nb_units):
		self.nb_layers = long(nb_layers)
		self.decay = decay
		self.optimizer_builder = optimizer_builder
		self.lr = lr
		self.batch_size = long(batch_size)
		self.epochs = long(epochs)
		self.dropout = dropout
		self.activation_function = activation_function
		self.nb_units = long(nb_units)
	def __str__(self):
		return "nb_layers:  "+str(self.nb_layers) + ";  "+ "decay:  "+str(self.decay) + ";  "+ "optimizer_builder:  "+str(self.optimizer_builder) + ";  "+ "lr:  "+str(self.lr) + ";  "+ "batch_size:  "+str(self.batch_size) + ";  "+ "epochs:  "+str(self.epochs) + ";  "+ "dropout:  "+str(self.dropout) + ";  "+ "activation_function:  "+str(self.activation_function) + ";  "+ "nb_units:  "+str(self.nb_units)

def create_search_space():
	search_space = {
		"nb_layers": hp.choice("nb_layers", [0,1,2,4,5,6]),
		'decay': hp.uniform('decay', 0, 0.9),
		'optimizer_builder': hp.choice('optimizer_builder', [SGD, Adam, RMSprop]),
		"lr": hp.uniform("learning_rate", 0.0001, 0.1),
		"batch_size": hp.uniform("batch_size", 16, 256),
		"epochs": hp.uniform("epoch", 1, 50),
		"dropout": hp.uniform("dropout", 0,0.5),
		"activation_function": hp.choice("activation_function", ["sigmoid", "relu", "tanh"]),
		"nb_units": hp.uniform("nb_units", 16, 128)
	}

	return search_space


def create_model_single(MAX_SEQ_LEN, hyperparameter):
    LSTM_DIM_SIZE = hyperparameter.nb_units
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

    # def concat():
    #     return np.concatenate((np.array(slist_param)*-1, np.array(rlist_param)))
    #     #TODO in the paper say we tried concatenating the sent and the received it gives 10% less accuracy

    return alternate_received_and_sent()

def build_burst_feature(data_folder):
    """Build a second feature consisting of the packet length sequences split in bursts."""
    flist = os.listdir(data_folder)
    features = []
    burst_features = []
    labels = []
    MAX_SEQ_LEN = 0
    for fname in flist:
        print(fname)
        with open(data_folder + fname) as f:
            data_dict = json.loads(f.read())
            for k, v in data_dict.items():
                new_list = make_single_feature(v['sent'], v['received'], v['order'])
                if len(new_list) > MAX_SEQ_LEN:
                    MAX_SEQ_LEN = len(new_list)
                features.append(new_list)
                labels.append(int(k[:-5]))
    #features = pad_sequences(features, dtype="float64", maxlen=MAX_SEQ_LEN)
    #burst_features = get_bursts(features)

    # Transform the feature vector in a vector of bursts
    #MAX_SEQ_LEN = 0
    for f in features:
        new_burst = ngrams_bursts(np.asarray(f))
        if len(new_burst) > MAX_SEQ_LEN:
            MAX_SEQ_LEN = len(new_burst)
        burst_features.append(new_burst)

    # Pad the result to have uniformly shaped inputs
    burst_features = pad_sequences(burst_features, dtype="float64", maxlen=MAX_SEQ_LEN)
    
    # Scale the bursts sequence between -1 and 1
    # burst_features = np.array(burst_features)
    # max_burst = np.max(burst_features)
    # min_burst = np.min(burst_features)
    # burst_features = np.float64(burst_features)
    # for b_f in burst_features:
    #     for elem in b_f:
    #         elem = elem / max_burst if elem > 0 else elem / min_burst

    print('Features shape: {}\t Bursts shape: {}'.format(np.array(features).shape, np.array(burst_features).shape))

    return np.array(burst_features), np.array(labels), MAX_SEQ_LEN

	def build_burst_feature(data_folder):
    """Build a second feature consisting of the packet length sequences split in bursts."""
    flist = os.listdir(data_folder)
    features = []
    burst_features = []
    labels = []
    MAX_SEQ_LEN = 0
    for fname in flist:
        print(fname)
        with open(data_folder + fname) as f:
            data_dict = json.loads(f.read())
            for k, v in data_dict.items():
                new_list = make_single_feature(v['sent'], v['received'], v['order'])
                if len(new_list) > MAX_SEQ_LEN:
                    MAX_SEQ_LEN = len(new_list)
                features.append(new_list)
                labels.append(int(k[:-5]))
    #features = pad_sequences(features, dtype="float64", maxlen=MAX_SEQ_LEN)
    #burst_features = get_bursts(features)

    # Transform the feature vector in a vector of bursts
    #MAX_SEQ_LEN = 0
    for f in features:
        new_burst = ngrams_bursts(np.asarray(f))
        if len(new_burst) > MAX_SEQ_LEN:
            MAX_SEQ_LEN = len(new_burst)
        burst_features.append(new_burst)

    # Pad the result to have uniformly shaped inputs
    burst_features = pad_sequences(burst_features, dtype="float64", maxlen=MAX_SEQ_LEN)
    
    # Scale the bursts sequence between -1 and 1
    # burst_features = np.array(burst_features)
    # max_burst = np.max(burst_features)
    # min_burst = np.min(burst_features)
    # burst_features = np.float64(burst_features)
    # for b_f in burst_features:
    #     for elem in b_f:
    #         elem = elem / max_burst if elem > 0 else elem / min_burst

    print('Features shape: {}\t Bursts shape: {}'.format(np.array(features).shape, np.array(burst_features).shape))

    return np.array(burst_features), np.array(labels), MAX_SEQ_LEN



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
                #     MAX_SEQ_LEN = len(rec_list)
                #features.append(rec_list) #Currently trying only with received lengths
                if len(new_list) > MAX_SEQ_LEN:
                    MAX_SEQ_LEN = len(new_list)
                features.append(new_list)
                labels.append(int(k[:-5]))
    features = pad_sequences(features, dtype="float64", maxlen=MAX_SEQ_LEN)
    return np.array(features), np.array(labels), MAX_SEQ_LEN


DataInfo = collections.namedtuple("DataInfo", "features labels max_len")


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
    fit_return = model.fit(X_train, y_train, batch_size=hyperparameter.batch_size, callbacks =[EarlyStopping(min_delta=0.03, patience=1) ],epochs=hyperparameter.epochs, validation_split= 0.05, shuffle= 'batch')

    score = model.evaluate(X_test, y_test)

    print("accuracy is", score[1])
    return score[1]


def create_sequence(min_val, max_val, number_steps):
    i = 0
    sequence = []
    step_size = (max_val - min_val) / number_steps
    while i < number_steps:
        sequence.append(min_val + step_size * i)
        i+=1 
    return sequence



def create_possible_hyperparameters():
	number_steps = 5

	decays = create_sequence(0, 0.9, number_steps)

	nb_units = create_sequence(16, 128, number_steps)
	optimizer_builders = [SGD, Adam, RMSprop]
	learning_rates = create_sequence(0.0001, 0.1, number_steps)

	batch_sizes = create_sequence(16, 256, number_steps)	
	possible_epochs = create_sequence(1, 50, number_steps)
	possible_nb_layers = [0,1,2,4,5,6] 
	dropouts = create_sequence(0, 0.5, number_steps)
	
	activation_functions = ["sigmoid", "relu", "tanh"] 

	
	cartesian_prod_result = itertools.product(possible_nb_layers, decays, optimizer_builders, learning_rates, batch_sizes, 
		possible_epochs, dropouts, activation_functions, nb_units)
	hyperparameters = []
	for hyperparameter_tuple in cartesian_prod_result:
		hyperparameters.append(Hyperparameter(*hyperparameter_tuple))

	print("There are {0} possible hyperparameters\n\n".format(len(hyperparameters)))
	return hyperparameters
	
	



def classify_LSTM(NUM_CLASSES, NUM_DAYS):

	def create_model_single(MAX_SEQ_LEN, hyperparameter):
		LSTM_DIM_SIZE = long(hyperparameter.nb_units)
		model = Sequential()
		model.add(Reshape((1, MAX_SEQ_LEN), input_shape=(MAX_SEQ_LEN,1)))

		for i in range(long(hyperparameter.nb_layers)):
			try:
				model.add(LSTM(
					units=LSTM_DIM_SIZE,
					activation=hyperparameter.activation_function,
					dropout=hyperparameter.dropout,
					return_sequences=True
				))
			except Exception:
				pdb.set_trace()
				
			
			
		
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


	def find_mean_and_standard_derivation(xs):
		as_single_list = np.reshape(xs, (-1))	
		mean = np.mean(as_single_list)
		standard_derivation = np.std(as_single_list)
		return mean, standard_derivation

	def standardize(xs, mean, standard_derivation):	
		return (xs - mean) / standard_derivation


	def single_feature(dataInfo, result_file_updater, params):
		args = []
		fields = inspect.getargspec(Hyperparameter.__init__).args[1:]
		for field in fields:
			args.append(params.get(field))
		hyperparameter = Hyperparameter(*args)
			
		features, olabels, max_len = dataInfo
		
		h=hpy()
		print(h.heap())

		features = np.reshape(features, [features.shape[0], max_len, 1]) #Add a dimension so keras is happy
		X_train, X_test, y_train, y_test = train_test_split(features, olabels, test_size=.2)#shuffles the data by default
		y_train = np.reshape(y_train, [len(y_train), 1])


		model = create_model_single(max_len, hyperparameter)
		print(model.summary())
		fit_return = model.fit(X_train, y_train, batch_size=long(hyperparameter.batch_size), callbacks =[EarlyStopping(min_delta=0.03, patience=1) ],epochs=long(hyperparameter.epochs), validation_split= 0.05, shuffle= 'batch')

		score = model.evaluate(X_test, y_test)

		print("accuracy is", score[1])
		result_file_updater(hyperparameter, score[1])
		return score[1]



	# urls = []
	# url_file = "short_list_500"
	# with open(url_file) as f:
	# 	lines = f.readlines()
	# 	for line in lines:
	# 		urls.append(line.strip())
	
	np.random.seed(404) #SEED used in the shuffle of hyperparameters and by keras

	datadir = closed_world_foldername(NUM_CLASSES, NUM_DAYS)
	if not os.path.isdir(datadir):
		create_closed_world(NUM_CLASSES, NUM_DAYS)	


	def get_dated_file_name(prefix):
		now = datetime.datetime.utcnow()
		return "{0}_d{1}_{2}h_{3}m".format(prefix, now.day, now.hour, now.minute)


	LOGS_DIR = "../logs/"
	RESULTS_DIR = "../results/"
	def create_if_not_exists(direc):
		if not os.path.isdir(direc):
			os.makedirs(direc)
			
	for d in [LOGS_DIR, RESULTS_DIR]:
		create_if_not_exists(d)

	dataInfo = DataInfo(*get_data_single(datadir))



        for d in [LOGS_DIR, RESULTS_DIR]:
                create_if_not_exists(d)
 
	result_filename = get_dated_file_name("../results/result_file_LSTM"+str(NUM_CLASSES)+"classes_"+str(NUM_DAYS)+"days")
	log_file_name = get_dated_file_name(os.path.join(LOGS_DIR, "log_train"))
	result_filename = get_dated_file_name(os.path.join(RESULTS_DIR, result_filename))

	nb_tried = {"nb_tried": 0}
	score_to_hyperparam = {}
	def update_result_file(hyperparameter, accuracy):
		score_to_hyperparam.update({accuracy: hyperparameter})
		print("UPDATING RESULT FILE")
		sorted_keys = sorted(score_to_hyperparam, reverse=True)
		nb = nb_tried.get("nb_tried")
		with open(result_filename, "w") as f:
			lines = """Model with {0} classes, interval of days [0, {1}]\ntried the first {2} hyperparameters\naccuracy_score\thyperparameter\n""".format(NUM_CLASSES, NUM_DAYS, nb)
			for hyperparam in sorted_keys: 
				lines += "{0}\t{1}\n".format(hyperparam, score_to_hyperparam.get(hyperparam))
			f.write(lines)
			nb_tried.update({"nb_tried": nb+1})




	trials_obj = Trials()
	best = fmin(fn=(lambda params: -1 * single_feature(dataInfo, update_result_file, params)), 
		space=create_search_space(), 
		algo=tpe.suggest, 
		max_evals=100, 
		trials=trials_obj)
	print(best)

if __name__ == '__main__':
	num_classes = 50
	num_days = 10000
	classify_LSTM(num_classes, num_days)
