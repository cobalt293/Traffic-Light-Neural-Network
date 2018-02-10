import csv
from queue import Queue
import random
import numpy as np

class Car(object):
    def __init__(self, enter_time):
        self.enter_time = enter_time

    def calculate_wait_time(self, cur_traffic_op):
        """given the currrent traffic op count this function 
        returns the total wait time it has spent in the intersection"""
        return cur_traffic_op-self.enter_time


class TrafficLight(object):
    """The main classs that will simulate the traffic light
    and the intersection.  This will take in to account
    new entering cars, what lights are green and red, 
    how how many cars are waiting in each direction.  
    This simulation is run by calling the run_traffic_flow_op
    method."""
    def __init__(self, log_filename='log.csv'):
        
        # Cars waiting in each lane
        self.north_q = []
        self.south_q = []
        self.east_q = []
        self.west_q = []

        # Performance Metrics of each lane
        self.north_wait_times = [0]
        self.south_wait_times = [0]
        self.east_wait_times = [0]
        self.west_wait_times = [0]

        # Chance a new car will enter the lane per traffic_op
        self.chance_north_q = 0.4
        self.chance_south_q = 0.4
        self.chance_east_q = 0.4
        self.chance_west_q = 0.4

        # Traffic Light Signals
        self.north_light = 'red'
        self.south_light = 'red'
        self.east_light = 'red'
        self.west_light = 'red'
        self.total_traffic_ops = 0
        self.light_vector = [0, 0, 0, 0]   # [N, S, E, W]

        #  Logging
        self.log_filename = log_filename

        # Empty the log file and create the header
        with open(self.log_filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, 
                        delimiter=',',
                        quotechar='|',
                        quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([
                'total_traffic_ops',
                'north_light',
                'south_light',
                'east_light',
                'west_light',
                'north_queue_size',
                'north_current_wait_time',
                'south_queue_size',
                'south_current_wait_time',
                'east_queue_size',
                'east_current_wait_time',
                'west_queue_size',
                'west_current_wait_time',
            ])

    def _calculate_average_wait_time(self, lane):
        """lane is a pointer to one of the lanes in the instance.  calaulates
        the average wait time of that lane"""
        wait_times_of_each_car = []
        for car in lane:
            wait_times_of_each_car.append(car.calculate_wait_time(self.total_traffic_ops))
        return np.nan_to_num(np.mean(wait_times_of_each_car))

    def run_traffic_flow_op(self, log=True, auto_light=True):
        """Runs the simulation 1 step.  This means that 1 car will pass
        through the intersection for the lights that are green"""
        #  North --------------------------------------------------
        self.north_wait_times.append(self._calculate_average_wait_time(self.north_q))
        if self.north_light=='green' and len(self.north_q)>0:
            north_exit_car = self.north_q.pop(0)
        if random.random() <= self.chance_north_q:
            self.north_q.append(Car(self.total_traffic_ops))

        #  South --------------------------------------------------
        self.south_wait_times.append(self._calculate_average_wait_time(self.south_q))
        if self.south_light=='green' and len(self.south_q)>0:
            south_exit_car = self.south_q.pop(0)
        if random.random() <= self.chance_south_q:
            self.south_q.append(Car(self.total_traffic_ops))

        # East --------------------------------------------------
        self.east_wait_times.append(self._calculate_average_wait_time(self.east_q))
        if self.east_light=='green' and len(self.east_q)>0:
            east_exit_car = self.east_q.pop(0)
        if random.random() <= self.chance_east_q:
            self.east_q.append(Car(self.total_traffic_ops))

        # West --------------------------------------------------
        self.west_wait_times.append(self._calculate_average_wait_time(self.west_q))
        if self.west_light=='green' and len(self.west_q)>0:
            west_exit_car = self.west_q.pop(0)
        if random.random() <= self.chance_west_q:
            self.west_q.append(Car(self.total_traffic_ops))

        self.total_traffic_ops += 1
        # Change lights based on number of cars in each direction
        self.light_switch_threshold = 10
        if auto_light:
            ns_size = len(self.north_q) + len(self.south_q)
            ew_size = len(self.east_q) + len(self.west_q)

            if (ns_size - ew_size) > self.light_switch_threshold:
                self.set_lights(north='green', south='green', east='red', west='red')

            if (ew_size - ns_size) > self.light_switch_threshold:
                self.set_lights(north='red', south='red', east='green', west='green')

        # Logging
        if log:
            self._log_intersection_summary()


    def _log_intersection_summary(self):
        """writes out to the csv file the current characteristics of
        the intersection.  This includes the lights, queue size of each
        lane, and the total number of traffic ops that have passed."""
        with open(self.log_filename, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)

            csv_writer.writerow([
                self.total_traffic_ops,
                self.light_vector[0],
                self.light_vector[1],
                self.light_vector[2],
                self.light_vector[3],
                len(self.north_q),
                self.north_wait_times[-1],
                len(self.south_q),
                self.south_wait_times[-1],
                len(self.east_q),
                self.east_wait_times[-1],
                len(self.west_q),
                self.west_wait_times[-1],
            ])

    def set_lights(self, north='red', south='red', east='red', west='red'):
        """Sets the four traffic lights and updates the light_vector"""
        self.north_light = north
        self.south_light = south
        self.east_light = east
        self.west_light = west

        light_codes = ['red', 'green']
        self.light_vector = [
            light_codes.index(self.north_light),
            light_codes.index(self.south_light),
            light_codes.index(self.east_light),
            light_codes.index(self.west_light),  
        ]
    
    def set_traffic_pattern(self, north=0.5, south=0.5, 
                            east=0.5, west=0.5):
        """Set the chance that a car will enter each lane per traffic op.
        This is these are percents"""
        self.chance_north_q = north
        self.chance_south_q = south
        self.chance_east_q = east
        self.chance_west_q = west


    def show_intersection(self):
        intersection = """\
            ------------------------------------------    Traiffic Light Signals: %s
            | Intersection ||  |  ||                 |      North: %5s
            |              ||  |  ||                 |      South: %5s
            |              ||  |  ||                 |       East: %5s
            |______________||%2s|  ||_________________|       West: %5s
            |_______________        %2s_______________|
            |_____________%2s        _________________|
            |              ||  |%2s||                 |
            |              ||  |  ||                 |
            |              ||  |  ||                 |
            |              ||  |  ||                 |
            ------------------------------------------""" %(self.total_traffic_ops,
                                                            self.north_light,
                                                            self.south_light,
                                                            self.east_light,
                                                            self.north_q,
                                                            self.west_light,
                                                            self.west_q,
                                                            self.east_q,
                                                            self.south_q)
        print(intersection)
