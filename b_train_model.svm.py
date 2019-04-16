
from FailureModels.Utility import get_train_test_split
import os
import pandas as pd
import numpy as np
from sklearn import svm

FAILURE_MODEL = os.path.abspath('FailureModels/saved_model/svm.pickle')
TRAINING_LOG = os.path.abspath('data/training_data_primary.csv')

X_train, y_train, X_test, y_test = get_train_test_split(TRAINING_LOG, COLUMNS)

data = pd.read_csv(TRAINING_LOG)
model = svm.SVC(gamma='scale') 
model.fit(X,y)


