from FailureModels.LstmNet import LstmNet
from FailureModels.Utility import get_train_test_split
import os
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

FAILURE_MODEL = os.path.abspath('FailureModels/saved_model/model')
TRAINING_LOG = os.path.abspath('data/training_data_primary.csv')

COLUMNS = [
    'cars_north_lane',
    'cars_south_lane',
    'cars_east_lane',
    'cars_west_lane',
]

X_train, y_train, X_test, y_test = get_train_test_split(TRAINING_LOG, COLUMNS)

model = LstmNet(FAILURE_MODEL)
model.fit(X_train, y_train, X_test, y_test, n_epochs=2)

# Calculate performance scores 
y_test_pred = model.predict(X_test)
#y_test_scores = np.amax(model.predict(X_test), axis=1)

from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
model_precision_score = precision_score(y_test, y_test_pred)
print("Precision Score: ", model_precision_score)

model_recall_score = recall_score(y_test, y_test_pred)
print("Recall Score: ", model_recall_score)

model_f1_score = f1_score(y_test, y_test_pred)
print("F1 Score: ", model_f1_score)

# def plot_roc_curve(fpr, tpr, label=None):
#     plt.plot(fpr, tpr, linewidth=2, label=label)
#     plt.plot([0,1], [0,1], 'k--')
#     plt.axis([0,1,0,1])
#     plt.xlabel('False Pasitive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.show()

# fpr, tpr, thresholds = roc_curve(y_test, y_test_scores)

plot_roc_curve(fpr, tpr)