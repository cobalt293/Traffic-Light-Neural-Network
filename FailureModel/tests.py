import unittest

from utility import get_train_test_split


class TestUtility(unittest.TestCase):
    def test_get_train_test_split(self):
        LOG_FILE = "../training_log.csv"
        keep_columns = ['cars_north_lane','cars_south_lane','cars_east_lane','cars_west_lane']
        X_train, y_train, X_test, y_test = get_train_test_split(LOG_FILE, keep_columns)
        print(X_train)


if __name__ == '__main__':
    unittest.main(verbosity=3)
