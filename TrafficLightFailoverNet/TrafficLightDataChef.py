import pandas as pd
import numpy as np

class TrafficLightDataChef(object):
    def __init__(self, file_name, column_names):
        # Read in Data and look and formate into train test split
        raw_data = pd.read_csv('log.csv')

        data_size = len(raw_data)
        sample_length = 50
        num_inputs = 4
        data = raw_data[column_names].values
        print(data.shape)

        # Populate the training data and perform train test split
        batch_size = data_size - sample_length
        X = np.zeros((batch_size, sample_length, num_inputs))
        #y = np.zeros(batch_size)
        y = raw_data['north_light'].values

        for i in range(batch_size-1):
            X[i] = data[i:i+sample_length].reshape(sample_length,-1)

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