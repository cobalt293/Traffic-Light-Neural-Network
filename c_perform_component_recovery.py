import matplotlib.pyplot as plt
import os

from TrafficSimulator.Simulator import Simulator
from TrafficSimulator.TrafficProfile import TrafficProfile
from FailureModels.LstmNet import LstmNet

FAILURE_MODEL = os.path.abspath('FailureModels/saved_model/model')
TRAFIC_PROFILE_FILE = os.path.abspath('./data/traffic_profile.csv')
RECOVERY_RESULT_FILE = os.path.abspath('./data/recovery_data_lstm.csv')

traffic_p = TrafficProfile()
traffic_p.load_data_file(TRAFIC_PROFILE_FILE)

sim_failover = Simulator()
sim_failover.add_failure_model(LstmNet(FAILURE_MODEL))
# # Run the Failover this is assuming it's already trained
# # let the normal algo take care of the first 100 timesteps
c = 0
for new_cars in traffic_p.iter_timesteps():
    if c<50:
        sim_failover.run_timestep_primary(new_cars)
    else:
        sim_failover.run_timestep_failover(new_cars)
    c += 1
sim_failover.flush_states_to_log(RECOVERY_RESULT_FILE)

sim_failover.state_store[[
    'cars_north_lane',
    'cars_south_lane',
    'cars_east_lane',
    'cars_west_lane',
    'light_state_north']].plot()

plt.show()