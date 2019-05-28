import os
import pandas as pd
import numpy as np
import  xgboost as xgb
import pickle
import matplotlib.pyplot as plt

FAILURE_MODEL = os.path.abspath('FailureModels/saved_model/xgboost.pickle')
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


model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)


model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc",
eval_set=[(X_test, y_test)])

with open(FAILURE_MODEL, 'wb') as f:
    pickle.dump(model, f)


# Calculate performance scores 
y_test_pred = model.predict(X_test)
y_test_scores = np.amax(model.predict_proba(X_test), axis=1)

from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
model_precision_score = precision_score(y_test, y_test_pred)
print("Precision Score: ", model_precision_score)

model_recall_score = recall_score(y_test, y_test_pred)
print("Recall Score: ", model_recall_score)

model_f1_score = f1_score(y_test, y_test_pred)
print("F1 Score: ", model_f1_score)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Pasitive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_test_scores)

plot_roc_curve(fpr, tpr)