

###--------------------------------------------------------------------------------------
### Create First Batch of Training Data
# Create the Traffic Light Simulator
from TrafficSimulator.Core import TrafficLight
import random
tl = TrafficLight()

tl.set_traffic_pattern(north=0.60, south=0.60, east=0.15, west=0.15)
tl.set_lights(north='green', south='green', east='red', west='red')

# Create the first set of training data for the Model
for i in range(400):
    tl.run_traffic_flow_op()



###--------------------------------------------------------------------------------------
### Create LSTM Model, format training data, and Train the model
# Instantiate the Data Generator class to create the training and testing data
from TrafficLightFailoverNet.TrafficLightDataChef import TrafficLightDataChef
training_data_formater = TrafficLightDataChef('log.csv', 'north_light')
X_train, y_train, X_test, y_test = training_data_formater.get_train_test_split()

# Instantiate the model 
import os
from datetime import datetime
log_dir = os.path.dirname(os.path.abspath(__file__))
log_dir += "/model/"+datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

from TrafficLightFailoverNet.TrafficLightNeuralNet import TrafficLightNeuralNet
model = TrafficLightNeuralNet(log_dir)
model.train(X_train, y_train, X_test, y_test, n_epochs=500)



# ###--------------------------------------------------------------------------------------
# ### Predict the next state of the lights and run the simulator based on the LSTM's Output
# # Have the model predict the next value of the lights
import pandas as pd
for i in range(200):
    X_pred = pd.read_csv('log.csv')['north_light'][-100:].values.reshape(1,100,1)
    y_pred = model.predict(X_pred)

    print(y_pred)

    if y_pred[0] == 0:
        print("0", y_pred[0], type(y_pred[0]))
        tl.set_lights(north='red', south='red', east='green', west='green')
    else:
        tl.set_lights(north='green', south='green', east='red', west='red')
    tl.run_traffic_flow_op(auto_light=False)



# # Take the next light state and set the traffic light accordingly
# if y_pred is 1:
#     tl.set_lights(north='green', south='green', east='red', west='red')
# else:
#     tl.set_lights(north='red', south='red', east='green', west='green')

# tl.run_traffic_flow_op()
# tl.set_traffic_pattern(north=random.random(), 
#                     south=random.random(), 
#                     east=random.random(), 
#                     west=random.random())
