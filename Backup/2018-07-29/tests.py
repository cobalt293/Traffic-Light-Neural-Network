import unittest
from Core import TrafficLight

class MyTest(unittest.TestCase):
    def test_TrafficLight_init(self):
        tl = TrafficLight()
        tl.set_lights(north='green', south='green', east='red', west='red')
        self.assertEqual(tl.light_vector, [1,1,0,0])

if __name__ == '__main__':
    unittest.main()
