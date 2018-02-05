

###--------------------------------------------------------------------------------------
### Create First Batch of Training Data
# Create the Traffic Light Simulator
from TrafficSimulator.Core import TrafficLight
import random
tl = TrafficLight()

tl.set_traffic_pattern(north=0.60, south=0.60, east=0.15, west=0.15)
tl.set_lights(north='green', south='green', east='red', west='red')

# Create the first set of training data for the Model
for i in range(200):
    tl.run_traffic_flow_op()
    tl.set_traffic_pattern(north=random.random(), 
                           south=random.random(), 
                           east=random.random(), 
                           west=random.random()
                           )



###--------------------------------------------------------------------------------------
### Create LSTM Model, format training data, and Train the model


# Instantiate the Data Generator class to create the training and testing data
from TrafficLightFailoverNet.TrafficLightDataChef import TrafficLightDataChef
training_data_formater = TrafficLightDataChef('log.csv', 'north_light')
X_train, y_train, X_test, y_test = training_data_formater.get_train_test_split()
print(X_train.shape)

# # train the model
# Instantiate the model 
#model = LSTMClassifier()
# model.train(learning_rate=0.001, epochs=150)



# ###--------------------------------------------------------------------------------------
# ### Predict the next state of the lights and run the simulator based on the LSTM's Output
# # Have the model predict the next value of the lights
# X_pred = # Shape(1,100,1)
# y_pred = model.predict(X_pred)

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
