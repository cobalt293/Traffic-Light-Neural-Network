from FailureModel import LstmNet
from FailureModel import Utility
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

traffic_p = TrafficProfile(20000)

sim_normal = Simulator()

# Run the primary system
for new_cars in traffic_p.iter_timesteps():
    sim_normal.run_timestep_primary(new_cars)

sim_normal.flush_states_to_log(TRAINING_LOG)

X_train, y_train, X_test, y_test = Utility.get_train_test_split(TRAINING_LOG, COLUMNS)

failure_model = LstmNet(FAILURE_MODEL)
failure_model.fit(X_train, y_train, X_test, y_test, n_epochs=3)
