from TrafficSimulator.Core import TrafficLight
from time import sleep
import random

# Create the Traffic Light Simulator
tl = TrafficLight()

tl.set_traffic_pattern(north=0.60, south=0.60, east=0.15, west=0.15)
tl.set_lights(north='green', south='green', east='red', west='red')

for i in range(800):
    tl.run_traffic_flow_op()
    tl.set_traffic_pattern(north=random.random(), 
                           south=random.random(), 
                           east=random.random(), 
                           west=random.random()
                           )