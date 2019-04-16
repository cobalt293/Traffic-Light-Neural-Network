from TrafficSimulator.TrafficProfile import TrafficProfile
from TrafficSimulator.Simulator import Simulator
import os

TRAFIC_PROFILE_FILE = os.path.abspath('./data/traffic_profile.csv')
TRAINING_DATA_PRIMARY_FILE = os.path.abspath('./data/training_data_primary.csv')
TRAINING_DATA_STATIC_FILE = os.path.abspath('./data/training_data_static.csv')

## Create a traffic profile
traffic_p = TrafficProfile()
traffic_p.generate(300)
traffic_p.to_csv(TRAFIC_PROFILE_FILE)


## Run simulation primary mode and produce training data
simulation = Simulator()
for new_cars in traffic_p.iter_timesteps():
    simulation.run_timestep_primary(new_cars)
simulation.flush_states_to_log(TRAINING_DATA_PRIMARY_FILE)


## Run simulation in static mode and produce training data
simulation = Simulator()
for new_cars in traffic_p.iter_timesteps():
    simulation.run_timestep_static(new_cars)
simulation.flush_states_to_log(TRAINING_DATA_STATIC_FILE)