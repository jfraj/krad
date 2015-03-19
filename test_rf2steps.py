import unittest
from rf2steps import RandomForestModel

class TestRandomForestModel(unittest.TestCase):
    """
    Testing the methods of rf2steps.RandomForestModel
    """
    def test_result_has_two_values(self):
        rfmodel = RandomForestModel('Data/train_2013.csv', 10, False)
        rfmodel.prepare_data(rfmodel.df_full)
        self.assertEqual(rfmodel.df_full.shape[0], 10)

if __name__=='__main__':
    unittest.main()
