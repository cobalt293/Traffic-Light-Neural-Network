import numpy as np
import random
import csv

class TrafficProfile(object):

    def __init__(self, data_file=None):
        self.north_arrivals = None
        self.south_arrivals = None
        self.east_arrivals = None
        self.west_arrivals = None
        
        self.total_cars = None

        if data_file:
            self.load_data_file(data_file)

    def load_data_file(self, file):
        self.north_arrivals = []
        self.south_arrivals = []
        self.east_arrivals = []
        self.west_arrivals = []
        try:
            with open(file, 'r') as f:
                csv_reader = csv.DictReader(f, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for row in csv_reader:
                    self.north_arrivals.append(int(row['north_arrivals']))
                    self.south_arrivals.append(int(row['south_arrivals']))
                    self.east_arrivals.append(int(row['east_arrivals']))
                    self.west_arrivals.append(int(row['west_arrivals']))
        except Exception as e:
            print("Something went wrong")
            print(e)

    def generate(self, num_timesteps):
        """Generates the arrival times for each of the lanes.
        Final result of the arrive times"""
        self.north_arrivals = []
        self.south_arrivals = []
        self.east_arrivals = []
        self.west_arrivals = []
        self.total_cars = 0

        north_south = np.random.poisson(15)/50
        east_west = .5-north_south

        for i in range(num_timesteps):
            if i% 10==0:
                north_south = np.random.poisson(15)/50
                east_west = .5-north_south

            # Used to determine if a new car is added
            chance_token = random.random() 

            # North South
            if chance_token <= north_south:
                self.north_arrivals.append(1)
                self.south_arrivals.append(1)
                self.total_cars += 2
            else:
                self.north_arrivals.append(0)
                self.south_arrivals.append(0)

            # East West
            if chance_token <= east_west:
                self.east_arrivals.append(1)
                self.west_arrivals.append(1)
                self.total_cars += 2
            else:
                self.east_arrivals.append(0)
                self.west_arrivals.append(0)

    def get_arrivals_at_timestep(timestep):
        """returns a dict of the arrivals at a current timestep"""
        if timestep < len(self.north_arrivals)-1:
            return {
                'north': self.north_arrivals[timestep],
                'south': self.south_arrivals[timestep],
                'east': self.east_arrivals[timestep],
                'west': self.west_arrivals[timestep]
            }
        else:
            raise IndexError

    def iter_timesteps(self):
        for i in range(len(self.north_arrivals)):
            yield {
                'north': self.north_arrivals[i],
                'south': self.south_arrivals[i],
                'east': self.east_arrivals[i],
                'west': self.west_arrivals[i]
            }

    def to_csv(self, file):
        with open(file, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['north_arrivals', 'south_arrivals', 'east_arrivals', 'west_arrivals'])
            for n,s,e,w in zip(self.north_arrivals, self.south_arrivals, self.east_arrivals, self.west_arrivals):
                csv_writer.writerow([n,s,e,w])
            

