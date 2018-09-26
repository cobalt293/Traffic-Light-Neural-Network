import unittest
import os
import pathlib

from utility import get_train_test_split, generate_random_batches

class TestUtility(unittest.TestCase):
    def test_get_train_test_split(self):
        LOG_FILE = "../training_log.csv"
        keep_columns = ['cars_north_lane','cars_south_lane','cars_east_lane','cars_west_lane']
        X_train, y_train, X_test, y_test = get_train_test_split(LOG_FILE, keep_columns)
        self.assertEqual(len(X_train.shape), 3)

    def test_generate_random_batches(self):
        LOG_FILE = "../training_log.csv"
        keep_columns = ['cars_north_lane','cars_south_lane','cars_east_lane','cars_west_lane']
        X_train, y_train, X_test, y_test = get_train_test_split(LOG_FILE, keep_columns)
        batch_size = 10
        X_batches, y_batches = generate_random_batches(X_train, y_train, batch_size)

        self.assertEqual(X_batches[0].shape[0],10)

if __name__ == '__main__':
    unittest.main(verbosity=3)
