
from TrafficSimulator.models.TrafficProfile import TrafficProfile
from TrafficSimulator.models.Simulator import Simulator

## Create a traffic profile
traffic_p = TrafficProfile(100)



# ## Create the simulator and give the the traffic profile
sim_normal   = Simulator()
# sim_failover = Simulator()

# failover_model = LSTMFailover()

# Run the normal
for new_cars in traffic_p.iter_timesteps():
    sim_normal.run_timestep_primary(new_cars)
    #print(sim_normal)

sim_normal.flush_states_to_log('testlog.csv')

# # Run the Failover this is assuming it's already trained
# # let the normal algo take care of the first 100 timesteps
# for new_cars in traffic_p:
#     sim_failover.run_op_failover(new_cars)

# sim_failover.flush_log_to_file('./models/log/failover_operation.csv')





