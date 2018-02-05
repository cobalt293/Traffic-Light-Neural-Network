import pandas as pd
import numpy as np

class TrafficLightDataChef(object):
    def __init__(self, file_name, column_name):
        # Read in Data and look at the first 400 datapoints on the north light
        raw_data = pd.read_csv('log.csv')

        data_size = len(raw_data)
        sample_length = 100
        num_inputs = 1
        data = raw_data[column_name].values

        # Populate the training data and perform train test split
        batch_size = data_size - sample_length
        X = np.zeros((batch_size, sample_length, num_inputs))
        y = np.zeros(batch_size)

        for i in range(batch_size-1):
            X[i] = data[i:i+sample_length].reshape(sample_length,-1)
            y[i] = data[i+sample_length+1]

        randomized = np.arange(batch_size)
        np.random.shuffle(randomized)
        X = X[randomized]
        y = y[randomized].astype(int)

        # split into train and test
        train_stop = np.floor(len(X) * 0.8).astype(int) # the index where training data stops and testing data starts

        self.X_train = X[:train_stop]
        self.y_train = y[:train_stop]

        self.X_test = X[train_stop:]
        self.y_test = y[train_stop:]

    def get_train_test_split(self):
        return self.X_train, self.y_train, self.X_test, self.y_test