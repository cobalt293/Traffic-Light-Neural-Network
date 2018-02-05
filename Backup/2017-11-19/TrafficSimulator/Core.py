import csv
from queue import Queue
import random import uniform


class Car(object):
    """Represents an individual car that is weighting in the intersection.
    There are several additional variables that help track the performance
    of the interection."""
    def __init__(self, unique_id, op_on_enter=None):
        self.unique_id = unique_id
        self.op_on_enter = op_on_enter
        self.op_on_exit = None
        self.delay_ops = None

    def set_op_on_exit(self, op_on_exit):
        """sets the global op step when the car leaves the 
        intersection.  This also calculates the total number
        of ops that have passed while the cars is waiting 
        in the intersection."""
        self.op_on_exit
        if self.op_on_enter is not None:
            self.delay_ops = self.op_on_exit - self.op_on_enter

    def __str__(self):
        return """Car %s""" % self.unique_id

class IntersectionQueue(object):
    """Represents a direction of cars waiting to cross the 
    intersection"""
    def __init__(self, new_car_probability=1.0):
        self.q = Queue()
        self.new_car_probability = new_car_probability

    def add_car(self,car, enter_op=None):
        """Adds a car with the option to tag what op number
        the car entered the intersection"""
        self.q.put(car)

    def remove_car(self, op_on_exit):
        """removes the first car on the queue and returns up
        to the simulator for further analysis
        Args:
            op_on_exit (int): the global op step of when the
                              car leavs the intersection
        Return: 
            Car: the car that left the intersection
            None: there are no cars to leave the intersection
        """
        if not self.q.empty():
            car = self.q.get()
            car.set_op_on_exit(op_on_exit)
            return car
        else:
            return None

    def size(self):
        """Returns the size of the queue as an int
        
        Return:
            int: the size of the queue
        """
        return self.q.qsize()

class TrafficLight(object):
    """The main classs that will simulate the traffic light
    and the intersection.  This will take in to account
    new entering cars, what lights are green and red, 
    how how many cars are waiting in each direction.  
    This simulation is run by calling the run_traffic_flow_op
    method."""
    def __init__(self, log_filename='log.csv'):
        self.north_q = IntersectionQueue()
        self.north_q_new_car_chance = None
        self.south_q = IntersectionQueue()
        self.east_q = IntersectionQueue()
        self.west_q = IntersectionQueue()
        self.north_light = 'red'
        self.south_light = 'red'
        self.east_light = 'red'
        self.west_light = 'red'
        self.total_traffic_ops = 0
        self.unique_car_id_counter = 1
                          # [N, S, E, W]
        self.light_vector = [0, 0, 0, 0]
        self.log_filename = log_filename

        # Empty the log file
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
                'south_queue_size',
                'east_queue_size',
                'west_queue_size'
            ])

    def run_traffic_flow_op(self, log=True):
        """Runs the simulation 1 step.  This means that 1 car will pass
        through the intersection for the lights that are green"""
        cars_leaving_intersection = []
        if self.north_light == 'green':
            car = self.north_q.remove_car(self.total_traffic_ops)
            if car is not None:
                cars_leaving_intersection.append(car)
        if self.south_light == 'green':
            car = self.south_q.remove_car(self.total_traffic_ops)
            if car is not None:
                cars_leaving_intersection.append(car)
        if self.east_light == 'green':
            car = self.east_q.remove_car(self.total_traffic_ops)
            if car is not None:
                cars_leaving_intersection.append(car)
        if self.west_light == 'green':
            car = self.west_q.remove_car(self.total_traffic_ops)
            if car is not None:
                cars_leaving_intersection.append(car)
        self.total_traffic_ops += 1

        if log:
            self._log_intersection_summary()

        return cars_leaving_intersection

    def _log_intersection_summary(self):
        """writes out to the csv file the current characteristics of
        the intersection.  This includes the lights, queue size of each
        lane, and the total number of traffic ops that have passed."""
        with open(self.log_filename, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, 
                                    delimiter=',',
                                    quotechar='|',
                                    quoting=csv.QUOTE_MINIMAL)

            csv_writer.writerow([
                self.total_traffic_ops,
                self.light_vector[0],
                self.light_vector[1],
                self.light_vector[2],
                self.light_vector[3],
                self.north_q.size(),
                self.south_q.size(),
                self.east_q.size(),
                self.west_q.size(),

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

    def add_car_to_north_queue(self):
        """Add a new car to the queue"""
        c = Car(self.unique_car_id_counter)
        self.unique_car_id_counter += 1
        self.north_q.add_car(c)

    def add_car_to_south_queue(self):
        """Add a new car to the queue"""
        c = Car(self.unique_car_id_counter)
        self.unique_car_id_counter += 1
        self.south_q.add_car(c)

    def add_car_to_east_queue(self):
        """Add a new car to the queue"""
        c = Car(self.unique_car_id_counter)
        self.unique_car_id_counter += 1
        self.east_q.add_car(c)

    def add_car_to_west_queue(self):
        """Add a new car to the queue"""
        c = Car(self.unique_car_id_counter)
        self.unique_car_id_counter += 1
        self.west_q.add_car(c)

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
                                                            self.north_q.size(),
                                                            self.west_light,
                                                            self.west_q.size(),
                                                            self.east_q.size(),
                                                            self.south_q.size())
        print(intersection)