import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

FAILURE_LOG = os.path.abspath('failure_log.csv')
STATIC_SWITCH_LOG = os.path.abspath('static_switch_log.csv')
PRIMARY_LOG = os.path.abspath('primary_log.csv')

primary = pd.read_csv(PRIMARY_LOG)
static_switch = pd.read_csv(STATIC_SWITCH_LOG)
failure = pd.read_csv(FAILURE_LOG)

primary[['cars_north_lane','cars_east_lane']].plot()
static_switch[['cars_north_lane','cars_east_lane']].plot()
failure[['cars_north_lane','cars_east_lane']].plot()

plt.show()