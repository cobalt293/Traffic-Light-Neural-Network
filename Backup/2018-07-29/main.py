

###--------------------------------------------------------------------------------------
### Create First Batch of Training Data
# Create the Traffic Light Simulator
from TrafficSimulator.Core import TrafficLight
import random
tl = TrafficLight()

#Poisson distribution
import numpy as np
north_south = np.random.normal(loc=.25, scale=.08)
east_west = .5-north_south

tl.set_traffic_pattern(north=north_south, south=north_south, east=east_west, west=east_west)
tl.set_lights(north='green', south='green', east='red', west='red')

# Create the first set of training data for the Model
for i in range(30000):
    tl.run_traffic_flow_op()
    if i % 200:
        north_south = np.random.normal(loc=.25, scale=.08)
        east_west = .5-north_south
        tl.set_traffic_pattern(north=north_south, south=north_south, east=east_west, west=east_west)



###--------------------------------------------------------------------------------------
### Create LSTM Model, format training data, and Train the model
# Instantiate the Data Generator class to create the training and testing data
COLUMNS = [
    'north_queue_size',
    'south_queue_size',
    'east_queue_size',
    'west_queue_size',
]
LOG_FILENAME = 'log.csv'

from TrafficLightFailoverNet.TrafficLightDataChef import TrafficLightDataChef

training_data_formater = TrafficLightDataChef(LOG_FILENAME, COLUMNS)
X_train, y_train, X_test, y_test = training_data_formater.get_train_test_split()

# Instantiate the model 
import os
from datetime import datetime
log_dir = os.path.dirname(os.path.abspath(__file__))
log_dir += "/model/"+datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

from TrafficLightFailoverNet.TrafficLightNeuralNet import TrafficLightNeuralNet
model = TrafficLightNeuralNet(log_dir)
model.train(X_train, y_train, X_test, y_test, n_epochs=10000)



## ###--------------------------------------------------------------------------------------
## ### Predict the next state of the lights and run the simulator based on the LSTM's Output
# Have the model predict the next value of the lights
import pandas as pd
for i in range(600):
    # Pull from CSV, standardize, and predict
    X_pred = pd.read_csv('log.csv')[COLUMNS][-50:].values.reshape(1,50,4)
    X_pred = (X_pred - X_pred.mean(axis=1)) / X_pred.std(axis=1) 
    y_pred = model.predict(X_pred)

    # Set traffic light state based on output of Model
    if y_pred[0] == 0:
        tl.set_lights(north='red', south='red', east='green', west='green')
    else:
        tl.set_lights(north='green', south='green', east='red', west='red')
    tl.run_traffic_flow_op(auto_light=False)

    # Randomly change the distro of traffic
    if i % 200:
        north_south = np.random.normal(loc=.25, scale=.08)
        east_west = .5-north_south
        tl.set_traffic_pattern(north=north_south, 
                               south=north_south,
                               east=east_west,
                               west=east_west)

## Plot the results
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt_columns = [
    'north_current_wait_time',
    'south_current_wait_time',
    'east_current_wait_time',
    'west_current_wait_time',
]
graph_raw_data = pd.read_csv('log.csv')[-1200:]
graph_wait_times = graph_raw_data[plt_columns]

graph_north_light = graph_raw_data['north_light']
graph_simulator = graph_north_light.copy(deep=True)
graph_simulator[600:] = None
graph_neural_net = graph_north_light.copy(deep=True)
graph_neural_net[:600] = None

fig = plt.figure()


sub_plot_1 = fig.add_subplot(211)
sub_plot_1.set_title("""Average Lane Wait Time""")
sub_plot_1.plot(np.arange(len(graph_wait_times)),graph_wait_times)
sub_plot_1.set_ylabel("Average Wait Time per Intersection")
sub_plot_1.legend(plt_columns)

sub_plot_2 = fig.add_subplot(212)
sub_plot_2.set_title("""State of North Light before and after Neural Net takes over""")
sub_plot_2.plot(np.arange(len(graph_simulator)), graph_simulator, 'black') 
sub_plot_2.plot(np.arange(len(graph_neural_net)), graph_neural_net, 'black')

sub_plot_2.fill_between(np.arange(len(graph_simulator)), 0, graph_simulator, facecolor='blue', alpha=0.5)
sub_plot_2.fill_between(np.arange(len(graph_neural_net)),0,graph_neural_net, facecolor='orange', alpha=0.5)
sub_plot_2.set_ylabel("1 = Green Light    0 = Red Light")
sub_plot_2.set_xlabel("Simulator Timesteps")
plt.show()