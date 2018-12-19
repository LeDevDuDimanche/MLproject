import numpy as np
import collections
import os
import json
import itertools
import datetime
from classify_LSTM import DataInfo, get_data_single, create_sequence
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

LSTM_DIM_SIZE = 32
NUM_CLASSES = 20
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

def build_model(learn_params, max_length):
    """Build a CNN model using Keras"""
    model = Sequential()
    #model.add(Reshape((1, max_length)))
    
    # Build first layer
    model.add(Dropout(input_shape = (max_length, 1), rate = learn_params.dropout_rate))

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


def build_single_feature():
    pass

def build_burst_feature():
    pass

def run(data_info):
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
    features = np.reshape(features, [features.shape[0], max_len, 1]) #Add a dimension so keras is happy

    #labels = convert_labels(olabels)
    #labels = np.reshape(labels, [labels.shape[0], labels.shape[1], 1])

    X_train, X_test, y_train, y_test = train_test_split(features, olabels, test_size=.2) #shuffles the data by default
    y_train = np.reshape(y_train, [len(y_train), 1])


    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Build the model using the params
    model = build_model(learn_params, max_len)
    #model = build_linear_model(max_len) #linear model for testing



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

Hyperparameter = collections.namedtuple("Hyperparameter", "filters, kernel_size, padding, activation_function, strides, pool_size, lstm_units, dense_units, dense_activation_function, dropout_rate")

def create_possible_hyperparameters():
    """Returns all possible combinations of hyperparameters in the specified ranges."""
    number_steps = 5

    decays = create_sequence(0, 0.9, number_steps)

    optimizer_builders = [SGD, Adam, RMSprop]
    learning_rates = create_sequence(0.0001, 0.1, number_steps)

    batch_sizes = create_sequence(16, 256, number_steps)
    possible_epochs = create_sequence(1, 50, number_steps)
    possible_nb_layers = [0,1,2,4,5,6] 
    dropouts = create_sequence(0, 0.5, number_steps)
    
    activation_functions = ["sigmoid", "relu", "tanh"] 

    cartesian_prod_result = itertools.product(possible_nb_layers, decays, optimizer_builders, learning_rates, batch_sizes, 
        possible_epochs, dropouts, activation_functions)
    hyperparameters = []
    for hyperparameter_tuple in cartesian_prod_result:
        hyperparameters.append(Hyperparameter(*hyperparameter_tuple))

    print("There are {0} possible hyperparameters\n\n".format(len(hyperparameters)))
    return hyperparameters

def get_dated_file_name(prefix):
    now = datetime.datetime.utcnow()
    return "{0}_d{1}_{2}h_{3}m".format(prefix, now.day, now.hour, now.minute)

result_filename = get_dated_file_name("../results/result_file_CNN"+str(NUM_CLASSES)+"classes_"+str(NUM_DAYS)+"days")

def update_result_file(learn_params, acc):
    print("UPDATING RESULT FILE")
    with open(result_filename, "w") as f:
        f.write("accuracy_score\thyperparameters\n")
        f.write("{0}\t{1}\n".format(acc, learn_params.__dict__))

if __name__ == '__main__':
    np.random.seed(404) #SEED used in the shuffle of hyperparameters and by keras
    datadir = "../data_cw"+str(NUM_CLASSES)+"_day0_to_"+str(NUM_DAYS)+"/"
    data_info = DataInfo(*get_data_single(datadir))
    acc_score_value = run(data_info)

    # np.random.seed(404) #SEED used in the shuffle of hyperparameters and by keras
    # datadir = "../data_cw"+str(NUM_CLASSES)+"_day0_to_"+str(NUM_DAYS)+"/"
    # hyperparameters = create_possible_hyperparameters()
    # np.random.shuffle(hyperparameters)
    # hyperparameter_to_score = {}
    # dataInfo = DataInfo(*get_data_single(datadir))

    # i = 0
    #log_file_name = get_dated_file_name("../logs/log_train")
    # with open(log_file_name, "a") as log_file:
    #     for hyperparameter in hyperparameters:
    #         print("\nusing this hyperparameter: "+ str(hyperparameter)+"\n")

    #         accuracy_score_value = run(dataInfo, hyperparameter)
    #         if accuracy_score_value == None:
    #             print("=============DISCARDING MODEL============")
    #             continue

    #         result = {accuracy_score_value: hyperparameter}
    #         log_file.write("accuracy score: {0}, hyperparameter: {1}\n".format(accuracy_score_value, hyperparameter))
    #         hyperparameter_to_score.update(result)
    #         update_result_file(i+1)

    #         i+=1