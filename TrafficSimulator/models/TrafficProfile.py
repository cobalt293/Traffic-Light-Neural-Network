import numpy as np
import random

class TrafficProfile(object):
    
    def __init__(self, num_timesteps):
        self.north_arrivals = None
        self.south_arrivals = None
        self.east_arrivals = None
        self.west_arrivals = None
        
        self.total_cars = None

        self.generate(num_timesteps)

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
    
