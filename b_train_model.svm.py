import os
import pandas as pd
import numpy as np
from sklearn import svm
import pickle

FAILURE_MODEL = os.path.abspath('FailureModels/saved_model/svm.pickle')
TRAINING_LOG = os.path.abspath('data/training_data_primary.csv')

KEEP_COLUMNS = [
    'cars_north_lane',
    'cars_south_lane',
    'cars_east_lane',
    'cars_west_lane',
    'light_state_north']

raw_data = pd.read_csv(TRAINING_LOG)

data_size = len(raw_data)
sample_length = 50
num_inputs = len(KEEP_COLUMNS)
data = raw_data[KEEP_COLUMNS].values

# Populate the training data and perform train test split
X = data[:,:4]
y = data[:,4]

# Shuffle X and y
shuffle = np.arange(X.shape[0])
np.random.shuffle(shuffle)
X = X[shuffle]
y = y[shuffle].astype(int)

# split into train and test
train_stop = np.floor(len(X) * 0.8).astype(int) # the index where training data stops and testing data starts

X_train = X[:train_stop]
y_train = y[:train_stop]
X_test = X[train_stop:]
y_test = y[train_stop:]

model = svm.SVC(gamma='scale')
model.fit(X_train, y_train)

saved_model = pickle.dumps(model)


