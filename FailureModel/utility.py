import numpy as np
import pandas as pd

def get_train_test_split(log_file, keep_columns):
    """Reads the log file and keep the columns in keep_columns
    Shuffels the data and splits it into a training set and testing set
    return """
    raw_data = pd.read_csv(log_file)

    data_size = len(raw_data)
    sample_length = 50
    num_inputs = len(keep_columns)
    data = raw_data[keep_columns].values

    # Populate the training data and perform train test split
    batch_size = data_size - sample_length
    X = np.zeros((batch_size, sample_length, num_inputs))
    y = raw_data['light_state_north'].values

    for i in range(batch_size-1):
        X[i] = data[i:i+sample_length].reshape(sample_length,-1)
        X[i] = (X[i]-X[i].mean(axis=0)) / X[i].std(axis=0)

    # Shuffle X and y
    randomized = np.arange(batch_size)
    np.random.shuffle(randomized)
    X = X[randomized]
    y = y[randomized].astype(int)

    # split into train and test
    train_stop = np.floor(len(X) * 0.8).astype(int) # the index where training data stops and testing data starts

    X_train = X[:train_stop]
    y_train = y[:train_stop]
    X_test = X[train_stop:]
    y_test = y[train_stop:]

    return X_train, y_train, X_test, y_test

def generate_random_batches(X, y, batch_size):
    """shuffles the training set into batches of size batch_size
    will output:
    [[x_batch], [x_batch], .....], [[y_batch],[y_batch],....]"""
    
    # Shuffle X and y in the same way.
    shuffle = np.random.shuffle(np.arange(len(X)))
    X = X[shuffle]
    y = y[shuffle].astype(int)

    cursor = batch_size
    X_batches = []
    y_batches = []
    while cursor < len(X):  # add to the batches lists 
        X_batches.append(X[cursor-batch_size: cursor])
        y_batches.append(y[cursor-batch_size: cursor])
        cursor += batch_size
    
    return X_batches, y_batches