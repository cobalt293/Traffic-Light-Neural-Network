from FailureModel.LstmNet import FailureModel
from FailureModel.DataChef import DataChef

import os

FAILURE_MODEL = os.path.abspath('FailureModel/saved_model')
TRAINING_LOG = os.path.abspath('primary_log.csv')

print(FAILURE_MODEL)
COLUMNS = [
    'cars_north_lane',
    'cars_south_lane',
    'cars_east_lane',
    'cars_west_lane',
]

training_data_formater = DataChef(TRAINING_LOG, COLUMNS)
X_train, y_train, X_test, y_test = training_data_formater.get_train_test_split()

failure_model = FailureModel(FAILURE_MODEL)
failure_model.train(X_train, y_train, X_test, y_test, n_epochs=50)
