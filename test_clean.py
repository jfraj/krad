import unittest

import pandas as pd


class TestGetRadarLength(unittest.TestCase):
    ## Testing different cases of radar combinations
    
    def test_result_has_two_values(self):
        from clean import getRadarLength
        self.assertEqual(getRadarLength(""), (0,))

    def test_one_radar(self):
        from clean import getRadarLength
        self.assertEqual(getRadarLength("1"), (1,))

    def test_one_radar_two_points(self):
        from clean import getRadarLength
        self.assertEqual(getRadarLength("1 1"), (2,))

    def test_two_radars(self):
        from clean import getRadarLength
        self.assertEqual(getRadarLength("1 2"), (1, 1))

    def test_three_radars(self):
        from clean import getRadarLength
        self.assertEqual(getRadarLength("1 2 3"), (1, 1, 1))


class TestSeparateListInColumn(unittest.TestCase):
    ## Testing how clean.separate_listInColumn is use to create columns
    def test_full(self):
        from clean import separate_listInColumn
        data = {
            'a': ['5 4 3 2 1', '5 4 3 7 1', '6 7 7', '3 5 6 1'],
            'b': [(3, 2), (2, 3), (1, 2), (4, 0)],
            }
        df = pd.DataFrame(data)
        r1, r2 = (
            zip(*df[['b', 'a']]
            .apply(separate_listInColumn, axis=1))
            )
        self.assertEqual(r1, (
            [5.0, 4.0, 3.0],
            [5.0, 4.0],
            [6.0],
            [3.0, 5.0, 6.0, 1.0]
            ))
        self.assertEqual(r2, (
            [2.0, 1.0],
            [3.0, 7.0, 1.0],
            [7.0, 7.0],
            [],
            ))


if __name__ == '__main__':
    unittest.main()
