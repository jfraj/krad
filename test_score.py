import unittest
from score import heaviside, kaggle_metric
import numpy as N


class TestHeaviside(unittest.TestCase):
    '''
    Compares the heaviside function from the score module
    with a heaviside defined with alternate coding
    '''
    
    def alternative_heaviside(self,r):
        x = N.arange(70)
        sign = N.where( x-r >= 0., 1., -1.)
        h = 0.5*(sign+1.)
        return h

    def get_norm_difference(self,h_module,h_test):
        norm_difference = N.linalg.norm(h_module-h_test)
        return norm_difference 

    def assert_heaviside(self,r):
        h_test   = self.alternative_heaviside(r)
        h_module = heaviside(r)
        error    = self.get_norm_difference(h_module,h_test)

        self.assertAlmostEqual(error , 0.)

    def test_heaviside(self):
        list_r = [-1., 0., 1., 10., 70.]
        for r in list_r:
            self.assert_heaviside(r)

class TestMetric(unittest.TestCase):
    '''
    Testing score calculation
    '''

    def test_kaggle_metric_same_inputs(self):
        predictions = N.random.random([100,70])
        error = kaggle_metric(predictions ,predictions)

        self.assertAlmostEqual(error , 0.)

    def test_kaggle_metric_one_value(self):

        predictions = [  1. ]
        exact_values= [  0. ]

        expected_score = 1./70.
        computed_score = kaggle_metric(predictions ,exact_values)

        self.assertAlmostEqual(expected_score, computed_score)
 
        
if __name__ == '__main__':
    unittest.main()
