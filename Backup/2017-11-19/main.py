from TrafficSimulator.Core import TrafficLight
from time import sleep

# Create the Traffic Light Simulator
tl = TrafficLight()

tl.set_lights(north='green', south='green', east='red', west='red')
for i in range(16):
    if i % 3 == 0:
        tl.add_car_to_north_queue()
        tl.add_car_to_south_queue()
        tl.add_car_to_east_queue()
        tl.add_car_to_west_queue()
    tl.run_traffic_flow_op()

tl.set_lights(north='red', south='red', east='green', west='green')
for i in range(16):
    if i % 3 == 0:
        tl.add_car_to_north_queue()
        tl.add_car_to_south_queue()
        tl.add_car_to_east_queue()
        tl.add_car_to_west_queue()
    tl.run_traffic_flow_op()

tl.set_lights(north='green', south='green', east='red', west='red')
for i in range(16):
    if i % 3 == 0:
        tl.add_car_to_north_queue()
        tl.add_car_to_south_queue()
        tl.add_car_to_east_queue()
        tl.add_car_to_west_queue()
    tl.run_traffic_flow_op()

tl.set_lights(north='red', south='red', east='green', west='green')
for i in range(16):
    if i % 3 == 0:
        tl.add_car_to_north_queue()
        tl.add_car_to_south_queue()
        tl.add_car_to_east_queue()
        tl.add_car_to_west_queue()
    tl.run_traffic_flow_op()