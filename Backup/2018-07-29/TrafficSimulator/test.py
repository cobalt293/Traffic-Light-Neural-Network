from Core import TrafficLight, Car

tl = TrafficLight()

tl.set_traffic_pattern(north=0.4, south=0.4, east=0.25, west=0.25)
tl.set_lights(north='green', south='green', east='red', west='red')

for i in range(200):
    tl.run_traffic_flow_op()

# queue = [Car(2), Car(6)]
# tl.total_traffic_ops = 10
# print(tl._calculate_average_wait_time(queue))