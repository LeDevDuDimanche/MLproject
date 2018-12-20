import numpy as np
import collections
import os
import json
import itertools
import datetime
from classify_LSTM import DataInfo, get_data_single, make_single_feature
from utils.util import get_bursts, ngrams_bursts
from keras import metrics
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import LSTM, Dense, Dropout, Concatenate, Input, Conv1D, MaxPooling1D, Flatten, Reshape
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, np_utils
from keras import regularizers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

BATCH_SIZE = 16
EPOCHS = 6
BURST = False

LSTM_DIM_SIZE = 32
NUM_CLASSES = 100
NUM_DAYS = 10000
#NUM_FEATURES = 2


MAX_DESCENDING = 703 
MAX_ASCENDING = 144

class LearnParams:
    def __init__(self, filters, kernel_size, padding, activation_function, strides, pool_size, lstm_units, dense_units, dense_activation_function, dropout_rate):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation_function = activation_function
        self.strides = strides
        self.pool_size = pool_size
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dense_activation_function = dense_activation_function
        self.dropout_rate = dropout_rate

def build_model(learn_params, input_shape):
    """Build a CNN model using Keras"""
    model = Sequential()
    #model.add(Reshape((1, max_length)))
    
    # Build first layer
    model.add(Dropout(input_shape = input_shape, rate = learn_params.dropout_rate))

    # Build the rest of the layers
    model.add(Conv1D(
        filters = learn_params.filters,
        kernel_size = learn_params.kernel_size,
        padding = learn_params.padding,
        activation = learn_params.activation_function,
        strides = learn_params.strides
        #input_shape = (max_length, 1)
    ))

    # Add a MaxPooling layer
    model.add(MaxPooling1D(
        pool_size = learn_params.pool_size,
        padding = learn_params.padding
    ))

    # Add another Conv layer
    model.add(Conv1D(
        filters = learn_params.filters,
        kernel_size = learn_params.kernel_size,
        padding = learn_params.padding,
        activation = learn_params.activation_function,
        strides = learn_params.strides
    ))

    # Add another Maxpooling layer
    model.add(MaxPooling1D(
        pool_size = learn_params.pool_size,
        padding = learn_params.padding
    ))

    # Add an LSTM layer
    model.add(LSTM(units = learn_params.lstm_units))


    # model.add(Flatten())

    # Add a last Dense layer
    model.add(Dense(
        units = learn_params.dense_units,
        activation = learn_params.dense_activation_function
    ))

    return model

def build_linear_model(max_length):
    """Build a simple linear model for testing purposes"""
    model = Sequential()
    #model.add(Reshape((max_length,), input_shape=(max_length,)))
    #model.add(Dense(units = NUM_CLASSES, activation = 'softmax'))
    model.add(Dense(units = NUM_CLASSES, activation = 'softmax', input_shape=(max_length,)))
    return model

def make_single_feature_no_scaling(slist, rlist, olist):
    #https://en.wikipedia.org/wiki/Feature_scaling
    def alternate_received_and_sent():
        newlist = []
        def treat_element(x):
            newlist.append(x)

        for item in olist[::-1]:
            if item == 1:
                treat_element(slist.pop())
            else:
                treat_element(rlist.pop() * -1)
        return newlist[::-1]

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

def run(data_info, burst = False, burst_features = None):
    """Perform a normal run by creating a model, then training it and evaluating the results."""

    # Define all parameters used for the current run
    filters = 32
    kernel_size = 5
    padding = 'valid'
    activation_function = 'relu'
    strides = 1
    pool_size = 4
    lstm_units = 128
    dense_units = NUM_CLASSES
    dense_activation_function = 'softmax'
    dropout_rate = 0.1
    learn_params = LearnParams(filters, kernel_size, padding, activation_function, strides, pool_size, lstm_units, dense_units, dense_activation_function, dropout_rate)

    # Build the features and the data that we will use. We reshape them by adding a dimension of value 1 to be able to feed them to conv1D.
    features, olabels, max_len = data_info
    if burst:
        double_features = np.zeros((features.shape[0], max_len, 2))
        #features = np.reshape(features, [features.shape[0], max_len])
        double_features[:,:,0] = features
        double_features[:,:,1] = burst_features

        print(double_features.shape)

        X_train, X_test, y_train, y_test = train_test_split(double_features, olabels, test_size=.2) #shuffles the data by default
        y_train = np.reshape(y_train, [len(y_train), 1])
    else:
        features = np.reshape(features, [features.shape[0], max_len, 1]) #Add a dimension so keras is happy
        X_train, X_test, y_train, y_test = train_test_split(features, olabels, test_size=.2) #shuffles the data by default
        #y_train = np.reshape(y_train, [len(y_train), 1])


    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Build the model using the params
    if burst:
        input_shape = (max_len, 2)
        model = build_model(learn_params, input_shape)
    else:
        input_shape = (max_len, 1)
        model = build_model(learn_params, input_shape)

    # Define the metrics and optimizer used to compile the model
    #metrics = ['accuracy']
    learning_rate = 0.0008
    decay = 0.0
    optimizer = RMSprop(lr = learning_rate, decay = decay)

    # Compile the model with the parameters defined above
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = optimizer,
        metrics = [metrics.sparse_categorical_accuracy]
    )
    print(model.summary())


    # Fit the model on our training data, then evaluate it and try to predict the correct labels
    model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 1)
    score = model.evaluate(X_test, y_test)
    y_pred = model.predict_classes(X_test)
    ilabels = y_pred
    print("correct labels were", y_test, "infered labels are", ilabels)
    res = accuracy_score(y_test, y_pred)
    print("accuracy is", res)
    update_result_file(learn_params, res)
    return res

def get_dated_file_name(prefix):
    now = datetime.datetime.utcnow()
    return "{0}_{1}d_{2}h_{3}m".format(prefix, now.day, now.hour, now.minute)

result_filename = get_dated_file_name("../results/result_file_CNN"+str(NUM_CLASSES)+"classes_"+str(NUM_DAYS)+"days_burst" + str(BURST))

def update_result_file(learn_params, acc):
    print("UPDATING RESULT FILE")
    with open(result_filename, "w") as f:
        f.write("accuracy_score\t{0}\n".format(acc))
        f.write("hyperparameters\t{0}, batch_size = {1}, number_of_epoch = {2}\n".format(learn_params.__dict__, BATCH_SIZE, EPOCHS))

if __name__ == '__main__':
    np.random.seed(404) #SEED used in the shuffle of hyperparameters and by keras
    datadir = "../data_cw"+str(NUM_CLASSES)+"_day0_to_"+str(NUM_DAYS)+"/"
    burst_features = None
    if BURST:
        data_info = DataInfo(*build_burst_feature(datadir))
        burst_features, _, _ = data_info

    data_info = DataInfo(*get_data_single(datadir))
    acc_score_value = run(data_info, burst = BURST, burst_features = burst_features)