import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
import pickle

from TrafficSimulator.Simulator import Simulator
from TrafficSimulator.TrafficProfile import TrafficProfile

FAILURE_MODEL_PICKLE = os.path.abspath('FailureModels/saved_model/random_forest.pickle')
TRAFIC_PROFILE_FILE = os.path.abspath('./data/traffic_profile.csv')
RECOVERY_RESULT_FILE = os.path.abspath('./data/recovery_data_random_forest.csv')

traffic_p = TrafficProfile()
traffic_p.load_data_file(TRAFIC_PROFILE_FILE)

sim_failover = Simulator()
with open(FAILURE_MODEL_PICKLE, 'rb') as f:
    failure_model = pickle.load(f)
sim_failover.add_failure_model(failure_model)
# # Run the Failover this is assuming it's already trained
# # let the normal algo take care of the first 100 timesteps
c = 0
for new_cars in traffic_p.iter_timesteps():
    if c<50:
        sim_failover.run_timestep_primary(new_cars)
    elif c>500:
        break
    else:
        sim_failover.run_timestep_failover_2d(new_cars)
    c += 1

sim_failover.flush_states_to_log(RECOVERY_RESULT_FILE)

sim_failover.state_store[[
    'cars_north_lane',
    'cars_south_lane',
    'cars_east_lane',
    'cars_west_lane',
    'light_state_north']].plot()

plt.show()
