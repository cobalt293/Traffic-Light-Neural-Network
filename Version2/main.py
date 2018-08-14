
from TrafficSimulator.TrafficProfile import TrafficProfile
from TrafficSimulator.Simulator import Simulator

from FailureModel.LstmNet import FailureModel

import os

FAILURE_MODEL = os.path.abspath('FailureModel/saved_model/model')
FAILURE_LOG = os.path.abspath('failure_log.csv')
PRIMARY_LOG = os.path.abspath('primary_log.csv')


## Create a traffic profile
traffic_p = TrafficProfile(10000)

# ## Create the simulator and give the the traffic profile
sim_normal   = Simulator()
sim_failover = Simulator()



# Run the primary system
for new_cars in traffic_p.iter_timesteps():
    sim_normal.run_timestep_primary(new_cars)

sim_normal.flush_states_to_log(PRIMARY_LOG)


# Run the Failover this is assuming it's already trained
# let the normal algo take care of the first 100 timesteps
for new_cars in traffic_p.iter_timesteps():
    sim_failover.run_timestep_primary(new_cars)

sim_failover.flush_states_to_log(FAILURE_LOG)





