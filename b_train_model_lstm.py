from FailureModels.LstmNet import LstmNet
from FailureModels.Utility import get_train_test_split
import os

FAILURE_MODEL = os.path.abspath('FailureModels/saved_model/model')
TRAINING_LOG = os.path.abspath('data/training_data_primary.csv')

COLUMNS = [
    'cars_north_lane',
    'cars_south_lane',
    'cars_east_lane',
    'cars_west_lane',
]

X_train, y_train, X_test, y_test = get_train_test_split(TRAINING_LOG, COLUMNS)

failure_model = LstmNet(FAILURE_MODEL)
failure_model.fit(X_train, y_train, X_test, y_test, n_epochs=2)

