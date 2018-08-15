from FailureModel.LstmNet import FailureModel
from FailureModel.DataChef import DataChef
from TrafficSimulator.TrafficProfile import TrafficProfile
from TrafficSimulator.Simulator import Simulator

import os

FAILURE_MODEL = os.path.abspath('FailureModel/saved_model/model')
TRAINING_LOG = os.path.abspath('training_log.csv')

print(FAILURE_MODEL)
COLUMNS = [
    'cars_north_lane',
    'cars_south_lane',
    'cars_east_lane',
    'cars_west_lane',
]

traffic_p = TrafficProfile(500)

sim_normal = Simulator()

# Run the primary system
for new_cars in traffic_p.iter_timesteps():
    sim_normal.run_timestep_primary(new_cars)

sim_normal.flush_states_to_log(TRAINING_LOG)

training_data_formater = DataChef(TRAINING_LOG, COLUMNS)
X_train, y_train, X_test, y_test = training_data_formater.get_train_test_split()

failure_model = FailureModel(FAILURE_MODEL)
failure_model.train(X_train, y_train, X_test, y_test, n_epochs=75)
