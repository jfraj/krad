"""Module for fitting single random forest classifier."""
# Standard
import numpy as N

# Ressources
import multiprocessing

# sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation

# this projects
from basemodel import BaseModel


class RandomForestClf(BaseModel):
    """Model using random forest classifier."""
    def __init__(self, train_data_fname=None, nrows='all', **kwargs):
        """Initialize the data frame."""
        self.rainClassifier = None
        super(RandomForestClf, self).__init__(train_data_fname, nrows, **kwargs)

    def cv_scores(self, col2fit, **kwargs):
        """Produce fit and score for the given parameters."""
        # parameters
        nestimators = kwargs.get('nestimator', 10)
        maxdepth = kwargs.get('nestimator', 10)

        # cleaning
        if not self.iscleaned:
            print 'Preparing the data...'
            self.clean_data(self.df_full, True, col2fit)

        # classifier
        self.rainClassifier = RandomForestClassifier(n_estimators=nestimators,
                                                     max_depth=maxdepth)

        # make test and train
        test_size = 0.3
        rnd_seed = 0
        features_train, features_test, target_train, target_test = train_test_split(
            self.df_full[col2fit].values, self.df_full['rain'].values,
            test_size=test_size, random_state=rnd_seed)
        print("\nFitting with max_depth={} and n_estimators={}...".format(maxdepth, nestimators))
        self.rainClassifier = RandomForestClassifier(n_estimators=nestimators, max_depth=maxdepth)
        self.rainClassifier.fit(features_train, target_train)

        ## Number of cpu to use
        ## Making sure there is one free unless there is only one
        njobs = max(1, int(0.75*multiprocessing.cpu_count()))
        print('\n\nValidating with njobs = {}\n...\n'.format(njobs))

        print 'Cross validating on {} rows with njobs={}...'.format(target_test.shape[0], njobs)

        scores = cross_validation.cross_val_score(self.rainClassifier, features_test,
                                                  target_test, cv=10, n_jobs=njobs)
        print scores
        print('\n\nCross validation accuracy:{} (+/- {})\n'.format(round(scores.mean(), 3), round(scores.std() / 2, 3)))


if __name__ == "__main__":
    a = RandomForestClf('Data/train_2013.csv', 7000)
    coltofit = ['Avg_Reflectivity', 'Range_Reflectivity', 'Nval',
                'Avg_DistanceToRadar', 'Avg_RadarQualityIndex',
                'Range_RadarQualityIndex',
                'Avg_RR1', 'Range_RR1', 'Avg_RR2', 'Range_RR2',
                'Avg_RR3', 'Range_RR3', 'Avg_Zdr', 'Range_Zdr',
                'Avg_Composite', 'Range_Composite', 'Avg_HybridScan',
                'Range_HybridScan', 'Avg_Velocity', 'Range_Velocity',
                'Avg_LogWaterVolume', 'Range_LogWaterVolume',
                'Avg_MassWeightedMean', 'Range_MassWeightedMean',
                'Avg_MassWeightedSD', 'Range_MassWeightedSD',
                'Avg_RhoHV', 'Range_RhoHV'
                ]
    a.cvscore(coltofit)
