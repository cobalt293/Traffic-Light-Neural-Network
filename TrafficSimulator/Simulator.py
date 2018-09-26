import csv
import numpy as np
import pandas as pd
import uuid

class Car(object):
    def __init__(self, enter_timestep):
        self.enter_timestep = enter_timestep
        self.id = uuid.uuid4()

    def calculate_wait_time(self, current_timestep):
        """given the currrent traffic op count this function 
        returns the total wait time it has spent in the intersection"""
        return current_timestep-self.enter_timestep

class Simulator(object):
    def __init__(self):
        self.COLUMNS = [
            'timestep',
            'cars_north_lane',
            'cars_south_lane',
            'cars_east_lane',
            'cars_west_lane',
            'light_state_north',
            'light_state_south',
            'light_state_east',
            'light_state_west',
            'north_avg_wait_time',
            'south_avg_wait_time',
            'east_avg_wait_time',
            'west_avg_wait_time'
        ]
        self.timestep = 0

        self.cars_north_lane = []
        self.cars_south_lane = []
        self.cars_east_lane = []
        self.cars_west_lane = []
        
        self.north_avg_wait_time = -1
        self.south_avg_wait_time = -1
        self.east_avg_wait_time = -1
        self.west_avg_wait_time = -1

        self.light_state_north = 1
        self.light_state_south = 1
        self.light_state_east = 0
        self.light_state_west = 0

        self.state_store = pd.DataFrame(columns=self.COLUMNS)

        self.failure_model = None

    def append_state_to_log(self):
        self.state_store.loc[self.timestep] = [
            self.timestep,
            len(self.cars_north_lane),
            len(self.cars_south_lane),
            len(self.cars_east_lane),
            len(self.cars_west_lane),
            self.light_state_north,
            self.light_state_south,
            self.light_state_east,
            self.light_state_west,
            self.north_avg_wait_time,
            self.south_avg_wait_time,
            self.east_avg_wait_time,
            self.west_avg_wait_time
        ]

    def flush_states_to_log(self, log_file):
        # with open(file_path, 'w', newline='') as log_file:
        #     csv_writer = csv.writer(log_file,
        #                             delimiter=',',
        #                             quotechar='|',
        #                             quoting=csv.QUOTE_MINIMAL)
        #     csv_writer.writerow([
        #         'timestep',
        #         'light_state_north',
        #         'light_state_south',
        #         'light_state_east',
        #         'light_state_west',
        #         'cars_north_lane',
        #         'cars_south_lane',
        #         'cars_east_lane',
        #         'cars_west_lane',
        #         'north_avg_wait_time',
        #         'south_avg_wait_time',
        #         'east_avg_wait_time',
        #         'west_avg_wait_time'
        #     ])

        #     for state in self.state_store:
        #         csv_writer.writerow([
        #             state['timestep'],
        #             state['light_state_north'],
        #             state['light_state_south'],
        #             state['light_state_east'],
        #             state['light_state_west'],
        #             state['cars_north_lane'],
        #             state['cars_south_lane'],
        #             state['cars_east_lane'],
        #             state['cars_west_lane'],
        #             state['north_avg_wait_time'],
        #             state['south_avg_wait_time'],
        #             state['east_avg_wait_time'],
        #             state['west_avg_wait_time']
        #         ])
        self.state_store.to_csv(log_file, index=False, columns=self.COLUMNS)

    def _calculate_average_wait_time(self, lane):
        """lane is a pointer to one of the lanes in the instance.  calaulates
        the average wait time of that lane"""
        wait_times_of_each_car = []
        for car in lane:
            wait_times_of_each_car.append(car.calculate_wait_time(self.timestep))
        return np.nan_to_num(np.mean(wait_times_of_each_car))

    def run_timestep_primary(self, new_traffic):
        """takes a dict of the new cars coming from each lane.  lets
        traffic flow one op and determins the state of the light.
        Finally appends to the state_log"""
        self._next_timestep(new_traffic)

        light_switch_threshold = 10

        # Update lights
        ns_size = len(self.cars_north_lane) + len(self.cars_south_lane)
        ew_size = len(self.cars_east_lane) + len(self.cars_west_lane)

        if (ns_size-ew_size) >= light_switch_threshold:
            self.light_state_north = 1
            self.light_state_south = 1
            self.light_state_east = 0
            self.light_state_west = 0
        elif (ew_size-ns_size) >= light_switch_threshold:
            self.light_state_north = 0
            self.light_state_south = 0
            self.light_state_east = 1
            self.light_state_west = 1
        self.append_state_to_log()

    def run_timestep_static(self, new_traffic):
        """Will Switch Lights at a static inverval"""
        self._next_timestep(new_traffic)
        if self.timestep % 30 == 0:
            # NORTH
            if self.light_state_north == 1:
                self.light_state_north = 0
            else:
                self.light_state_north = 1

            # SOUTH
            if self.light_state_south == 1:
                self.light_state_south = 0
            else:
                self.light_state_south = 1

            # EAST
            if self.light_state_east == 1:
                self.light_state_east = 0
            else:
                self.light_state_east = 1

            # WEST
            if self.light_state_west == 1:
                self.light_state_west = 0
            else:
                self.light_state_west = 1
        self.append_state_to_log()

    def run_timestep_failover(self, new_traffic):
        """use the failure model to determin what the 
        state of the traffic lights should be"""
        self._next_timestep(new_traffic)
        
        decision_data = self.state_store[['cars_north_lane', 'cars_south_lane', 'cars_east_lane', 'cars_west_lane']][-50:].values.reshape(-1,50,4)

        decision_data = (decision_data - decision_data.mean(axis=1)) / decision_data.std(axis=1)
        # print(decision_data)
        # print(decision_data.shape)

        ## failover models makes a decision as to what the light should be 
        decision = self.failure_model.predict(decision_data)
        print("decision: ", decision)

        if decision== 1:
            print("setting north light green")
            self.light_state_north = 1
            self.light_state_south = 1
            self.light_state_east = 0
            self.light_state_west = 0
        else:
            print("setting north light red")
            self.light_state_north = 0
            self.light_state_south = 0
            self.light_state_east = 1
            self.light_state_west = 1


        self.append_state_to_log()

    def _next_timestep(self, new_traffic):
        # Add the new cars coming into the intersection
        if new_traffic['north'] == 1:
            self.cars_north_lane.append(Car(self.timestep))
        if new_traffic['south'] == 1:
            self.cars_south_lane.append(Car(self.timestep))
        if new_traffic['east'] == 1:
            self.cars_east_lane.append(Car(self.timestep))
        if new_traffic['west'] == 1:
            self.cars_west_lane.append(Car(self.timestep))

        # have traffic flow on the lights that are green
        if self.light_state_north == 1 and len(self.cars_north_lane)>0:
            self.cars_north_lane.pop(0)
        
        if self.light_state_south == 1 and len(self.cars_south_lane)>0:
            self.cars_south_lane.pop(0)
        
        if self.light_state_east == 1 and len(self.cars_east_lane)>0:
            self.cars_east_lane.pop(0)
        
        if self.light_state_west == 1 and len(self.cars_west_lane)>0:
            self.cars_west_lane.pop(0)

        # Update the average wait time of each lane
        self.north_avg_wait_time = self._calculate_average_wait_time(self.cars_north_lane)
        self.south_avg_wait_time = self._calculate_average_wait_time(self.cars_south_lane)
        self.east_avg_wait_time = self._calculate_average_wait_time(self.cars_east_lane)
        self.west_avg_wait_time = self._calculate_average_wait_time(self.cars_west_lane)
        
        self.timestep += 1

    def add_failure_model(self, failure_model):
        """Adds a failure model to the simulation"""
        self.failure_model = failure_model

    def __str__(self):
        s = """Timestep: {timestep}
        Light States: {n_state}, {s_state}, {e_state}, {w_state} 
        Lane Sizes: {n_size}, {s_size}, {e_size}, {w_size}
        Wait Times: {n_wait}, {s_wait}, {e_wait}, {w_wait}"""
        return s.format(
            timestep= self.timestep,
            n_state = self.light_state_north,
            s_state = self.light_state_south,
            e_state = self.light_state_east,
            w_state = self.light_state_west,
            n_size = len(self.cars_north_lane),
            s_size = len(self.cars_south_lane),
            e_size = len(self.cars_east_lane),
            w_size = len(self.cars_west_lane),
            n_wait = self.north_avg_wait_time,
            s_wait = self.south_avg_wait_time,
            e_wait = self.east_avg_wait_time,
            w_wait = self.west_avg_wait_time,
        )