import numpy as np
import collections
import os
import json
import itertools
from classify_rnn import DataInfo, get_data_single, convert_labels, one_in_max_of_cols
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import LSTM, Dense, Dropout, Concatenate, Input, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, np_utils
from keras import regularizers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

BATCH_SIZE = 256
EPOCHS = 3

LSTM_DIM_SIZE = 32
NUM_CLASSES = 20
#NUM_FEATURES = 2


MAX_DESCENDING = 703 
MAX_ASCENDING = 144

class LearnParams:
    def __init__(self, filters, kernel_size, padding, activation_function, strides, pool_size, dense_units, dense_activation_function):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation_function = activation_function
        self.strides = strides
        self.pool_size = pool_size
        self.dense_units = dense_units
        self.dense_activation_function = dense_activation_function

def build_model(learn_params):
    """Build a CNN model using Keras"""
    model = Sequential()

    # Build first layer
    model.add(Conv1D(
        filters = learn_params.filters,
        kernel_size = learn_params.kernel_size,
        padding = learn_params.padding,
        activation = learn_params.activation_function,
        strides = learn_params.strides
    ))

    # Build the rest of the layers

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
    #model.add(LSTM(units = learn_params.units))

    # Add a last Dense layer
    model.add(Dense(
        units = learn_params.dense_units,
        activation = learn_params.dense_activation_function
    ))

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
    dense_units = 100
    dense_activation_function = 'softmax'
    learn_params = LearnParams(filters, kernel_size, padding, activation_function, strides, pool_size, dense_units, dense_activation_function)

    # Build the features and the data that we will use
    features, olabels, max_len = data_info
    features = np.reshape(features, [features.shape[0], max_len, 1]) #Add a dimension so keras is happy
    labels = convert_labels(olabels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.2)#shuffles the data by default

    # Build the model using the params
    model = build_model(learn_params)

    # Define the metrics and optimizer used to compile the model
    metrics = ['accuracy']
    learning_rate = 0.001
    decay = 0.0
    optimizer = RMSprop(lr = learning_rate, decay = decay)

    # Compile the model with the parameters defined above
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = optimizer,
        metrics = metrics
    )
    print(model.summary())


    # Fit the model on our training data, then evaluate it and try to predict the correct labels
    model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = EPOCHS, batch_size = BATCH_SIZE)
    score = model.evaluate(X_test, y_test)
    print("Score is", score)
    print("Predictions :", model.predict(X_test))
    y_pred = one_in_max_of_cols(model.predict(X_test).T).T
    ilabels = np.nonzero(y_pred)
    print("correct labels were", np.nonzero(y_test), "infered labels are", ilabels)
    res = accuracy_score(y_test, y_pred)
    return res

if __name__ == '__main__':
    datadir = "../data_cw20_day0_to_30/"
    data_info = DataInfo(*get_data_single(datadir))
    acc_score_value = run(data_info)