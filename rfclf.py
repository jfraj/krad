"""Module for fitting single random forest classifier."""
# Standard
import numpy as N
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Ressources
import multiprocessing

# sklearn
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix

# this projects
from basemodel import BaseModel
from score import kaggle_metric


class RandomForestClf(BaseModel):
    """Model using random forest classifier."""
    def __init__(self, train_data_fname=None, nrows='all', **kwargs):
        """Initialize the data frame."""
        self.rainClassifier = None
        super(RandomForestClf, self).__init__(train_data_fname, nrows, **kwargs)

    def prepare_data(self, df, verbose=False, var2prep='all'):
        """prepare self.df_full for fitting.

        var2prep is a list of variables that will be needed.
        This will save time by cleaning only the needed variables
        """
        if self.iscleaned:
            print('Data already cleaned')
            return

        self.clean_data(df, verbose, var2prep)

        # Add a category column rain/norain (1/0)
        # Might consider using a threshold i.e. rain if Expected > threshold
        if 'Expected' in df.columns.values:
            df['rain'] = df['Expected'].apply(lambda x: int(N.round(x, 2)))
            #df['rain'] = df['Expected']

        self.iscleaned = True


    def cv_scores(self, col2fit, **kwargs):
        """Produce fit and score for the given parameters."""
        # parameters
        nestimators = kwargs.get('nestimator', 10)
        maxdepth = kwargs.get('nestimator', 10)

        # cleaning
        if not self.iscleaned:
            print 'Preparing the data...'
            self.prepare_data(self.df_full, True, col2fit)

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

        scores = cross_validation.cross_val_score(self.rainClassifier,
                                                  features_test,
                                                  target_test, cv=5,
                                                  n_jobs=njobs)
        print scores
        print('\n\nCross validation accuracy:{} (+/- {})\n'.format(round(scores.mean(), 3), round(scores.std() / 2, 3)))

    def fitNscore(self, col2fit, **kwargs):
        """Produce fit and score for the given parameters."""
        # parameters
        nestimators = kwargs.get('nestimator', 100)
        maxdepth = kwargs.get('maxdepth', 20)

        # cleaning
        if not self.iscleaned:
            print 'Preparing the data...'
            self.prepare_data(self.df_full, True, col2fit)

        #print('TESTING: DROPPING<5')
        #self.df_full = self.df_full[self.df_full['rain'] > 5]
        #self.df_full['rain'].plot(kind='hist')
        #plt.show()
        #raw_input('ok...')

        # classifier
        self.rainClassifier = RandomForestClassifier(n_estimators=nestimators,
                                                     max_depth=maxdepth)
        #self.rainClassifier = RandomForestRegressor(n_estimators=nestimators,
        #                                             max_depth=maxdepth)

        test_size = 0.3## fraction kept for testing
        rnd_seed = 0## for reproducibility

        features_train, features_test, target_train, target_test = train_test_split(
            self.df_full[col2fit].values, self.df_full['rain'].values,
            test_size=test_size, random_state=rnd_seed)

        print('\nFitting with max_depth={} and n_estimators={}...'.format(maxdepth, nestimators))
        self.rainClassifier = RandomForestClassifier(n_estimators=nestimators, max_depth=maxdepth)
        #self.rainClassifier = RandomForestRegressor(n_estimators=nestimators, max_depth=maxdepth)
        self.rainClassifier.fit(features_train, target_train)

        # Predict on the rest of the sample
        print('\nPredicting...')
        predictions = self.rainClassifier.predict(features_test)

        # Get and print the score
        print '\nScoring...'
        score = kaggle_metric(N.round(predictions), target_test)
        score_pois = kaggle_metric(N.round(predictions), target_test, 'poisson')
        print '\n\nScore(heaviside)={}'.format(score)
        print '\nScore(poisson)={}\n\n'.format(score_pois)

        ord_idx = N.argsort(self.rainClassifier.feature_importances_)#Feature index ordered by importance
        for ifeaturindex in ord_idx[::-1]:
            print '{0} \t: {1}'.format(col2fit[ifeaturindex], round(self.rainClassifier.feature_importances_[ifeaturindex], 2))


        # Plots
        # confusion matrix
        cm = confusion_matrix(target_test.astype(int), predictions.astype(int))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, N.newaxis]
        cm_normalized = N.clip(cm_normalized, 0.0, 0.5)

        fig_cm = plt.figure()
        ax_cm = fig_cm.add_subplot(1,1,1)
        im_cm = ax_cm.imshow(cm_normalized, interpolation='nearest')
        plt.title('Confusion mtx, md={} ne={}'.format(maxdepth, nestimators))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        fig_cm.colorbar(im_cm)
        fig_cm.show()

        raw_input('press enter when finished...')

if __name__ == "__main__":
    a = RandomForestClf('Data/train_2013.csv', 5000)
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
                'Avg_RhoHV', 'Range_RhoHV',
                ]
    # Adding hydrometeor types
    # Some are merged because they have the same meaning
    hm_types = [0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
    coltofit.extend(["hm_{}".format(i) for i in hm_types])

    print coltofit
    #a.cvscore(coltofit)
    a.fitNscore(coltofit)
