"""Module for fitting single random forest classifier."""
# Standard
import numpy as N
import matplotlib.pyplot as plt
import pandas as pd

# Ressources
import multiprocessing

# sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

# this projects
from basemodel import BaseModel
from score import kaggle_metric, poisson_cumul
import solution


class RandomForestClf(BaseModel):

    """Model using random forest classifier."""

    def __init__(self, train_data_fname=None, nrows='all', **kwargs):
        """Initialize the data frame."""
        clf_pkl = kwargs.get('clf_pkl', False)
        if clf_pkl:
            print('\nUsing pickled clf from {}'.format(clf_pkl))
            self.rainClassifier = joblib.load(clf_pkl)
            self.fitted = True
            self.iscleaned = False
            return
        self.rainClassifier = None
        self.fitted = False
        super(RandomForestClf, self).__init__(train_data_fname, nrows, **kwargs)

    def prepare_data(self, df, verbose=False, var2prep='all', **kwargs):
        """prepare self.df_full for fitting.

        var2prep is a list of variables that will be needed.
        This will save time by cleaning only the needed variables
        """
        ignore_clean = kwargs.get('ignore_clean', False)
        if self.iscleaned and not ignore_clean:
            print('Data for classifier is already cleaned')
            return

        # Generic cleaning
        self.clean_data(df, verbose, var2prep, **kwargs)

        # Make a int column from expected
        if 'Expected' in df.columns.values:
            df['rain'] = df['Expected'].apply(lambda x: int(N.round(x, 2)))

        # Removing useless columns
        print('Removing useless columns...')
        to_keep = var2prep + ['Expected', 'rain', 'Id']
        for icol in df.columns:
            if icol not in to_keep:
                df.drop(icol, axis=1, inplace=True)

        self.iscleaned = True

    def prepareNsave(self, df, col2save, **kwargs):
        """Clean and prepare data and save pickle it."""
        save_name = kwargs.get('save_name', 'saved_clf/default.pkl')
        print('\nWill prepare and save the following column')
        print(col2save)
        print('Preparing the data...')
        self.prepare_data(df, True, col2save, **kwargs)
        print('Pickling dataframe...')
        df.to_pickle(save_name)
        print('Done saving dataframe in {}'.format(save_name))

    def set_classifier(self, **kwargs):
        """Set the classifier."""
        nestimators = kwargs.get('nestimator', 225)
        maxdepth = kwargs.get('maxdepth', 19)
        class_weight = "auto"

        print('\nClassifier max_depth={}'.format(maxdepth))
        print('n_estimators={}'.format(nestimators))
        print('class_weight={}'.format(class_weight))
        self.rainClassifier = RandomForestClassifier(n_estimators=nestimators,
                                                     max_depth=maxdepth,
                                                     class_weight=class_weight)

    def fitModel(self, values2fit, targets, **kwargs):
        """Fit the classifier."""
        if self.fitted:
            print('Already fitted...')
            return
        # Classifier
        self.set_classifier(**kwargs)

        print('Fitting on values with shape:')
        print(values2fit.shape)
        print('\nFitting...')
        self.rainClassifier.fit(values2fit, targets)
        self.fitted = True
        print('Done fitting!')

    def cv_scores(self, col2fit, **kwargs):
        """Produce fit and score for the given parameters."""
        # parameters
        nestimators = kwargs.get('nestimator', 10)
        maxdepth = kwargs.get('maxdepth', 10)
        class_weight = "auto"

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
        print("class_weight=")
        print(class_weight)
        self.rainClassifier = RandomForestClassifier(n_estimators=nestimators,
                                                     max_depth=maxdepth,
                                                     class_weight=class_weight)
        self.rainClassifier.fit(features_train, target_train)

        # Number of cpu to use
        # Making sure there is one free unless there is only one
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
        nestimators = kwargs.get('nestimator', 225)
        maxdepth = kwargs.get('maxdepth', 19)

        # cleaning
        if not self.iscleaned:
            print 'Preparing the data...'
            self.prepare_data(self.df_full, True, col2fit)

        test_size = 0.3  # fraction kept for testing
        rnd_seed = 0  # for reproducibility

        features_train, features_test, target_train, target_test =\
            train_test_split(self.df_full[col2fit].values,
                             self.df_full['rain'].values,
                             test_size=test_size,
                             random_state=rnd_seed)

        # Fit classifier
        self.fitModel(features_train, target_train, **kwargs)

        # Predict on the rest of the sample
        print('\nPredicting...')
        predictions = self.rainClassifier.predict(features_test)

        # Get and print the score
        print('\nScoring...')
        score = kaggle_metric(N.round(predictions), target_test)
        score_pois = kaggle_metric(N.round(predictions), target_test, 'poisson')
        print('\n\nScore(heaviside)={}'.format(score))
        print('\nScore(poisson)={}\n\n'.format(score_pois))

        # Feature index ordered by importance
        ord_idx = N.argsort(self.rainClassifier.feature_importances_)
        print("Feature ranking:")
        for ifeaturindex in ord_idx[::-1]:
            print('{0} \t: {1}'.format(col2fit[ifeaturindex], round(self.rainClassifier.feature_importances_[ifeaturindex], 2)))

        # Plots

        # Feature importances
        importances = self.rainClassifier.feature_importances_
        std = N.std([tree.feature_importances_ for tree in self.rainClassifier.estimators_],
             axis=0)
        indices = N.argsort(importances)[::-1]
        ordered_names = [ col2fit[i] for i in indices]

        fig_import = plt.figure(figsize=(10, 10))
        plt.title("Feature importances, md={} ne={}".format(maxdepth, nestimators))
        plt.barh(range(len(indices)), importances[indices],
                color="b", xerr=std[indices], align="center",ecolor='r')
        plt.yticks(range(len(indices)), ordered_names)
        plt.ylim([-1, len(indices)])
        plt.ylim(plt.ylim()[::-1])
        plt.subplots_adjust(left=0.22)
        fig_import.show()


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

    def fitNsave(self, col2fit, savename, **kwargs):
        """Fit and save the classifier"""
        # Preparing training data
        if not self.iscleaned:
            print('Preparing the data...')
            self.prepare_data(self.df_full, True, col2fit)
        self.fitModel(self.df_full[col2fit].values,
                      self.df_full['rain'].values, **kwargs)
        joblib.dump(self.rainClassifier, savename)
        print('classifier saved in {}'.format(savename))

    def submit(self, col2fit, **kwargs):
        """Create csv file for submission."""
        # Preparing training data
        if not self.iscleaned and not self.fitted:
            print('Preparing the data...')
            self.prepare_data(self.df_full, True, col2fit)

        # Fitting train data
        if not self.fitted:
            self.fitModel(self.df_full[col2fit].values,
                          self.df_full['rain'].values, **kwargs)
            print('\ndeleting training data')
            del(self.df_full)
            self.df_full = None

        # Predicting test data
        test_pickle = kwargs.get('test_pickle', None)
        if test_pickle is not None:
            print('Using pickled test sample {}'.format(test_pickle))
            df_test = pd.io.pickle.read_pickle(test_pickle)
        else:
            print('\nGetting and cleaning all test data...')
            df_test = pd.read_csv('Data/test_2014.csv')
            #df_test = pd.read_csv('Data/test_2014.csv', nrows=2000)
            self.prepare_data(df_test, True, col2fit, ignore_clean=True)

        # Divide the predictions into chunks of subrows
        subrows = 10000
        # Create a list of rows to s
        all_ranges = range(1, int(len(df_test)/subrows) + 1)
        all_ranges = [subrows*x for x in all_ranges]
        all_ranges[-1] = len(df_test)  # Last element extended to the last row
        last_row = 0
        open_type = 'w'
        for irange in all_ranges:
            ilist_id = df_test[last_row:irange]['Id'].values
            ival2predict = df_test[last_row:irange][col2fit].values
            print('\nprediction rows {}-{}'.format(last_row, irange))
            ipredictions = self.rainClassifier.predict(ival2predict)
            isub_data = N.array(map(poisson_cumul, N.round(ipredictions)))
            print('writing isubmission data...')
            solution.generate_submission_file(ilist_id, isub_data,
                                              open_type=open_type)
            open_type = 'a'
            last_row = irange
        #list_id = df_test[:sel_rows]['Id'].values
        #values2predict = df_test[:sel_rows][col2fit].values
        #print('\npredicting...')
        #predictions = self.rainClassifier.predict(values2predict[:])
        print('\ndeleting test data')
        del(df_test)

        # Creating prediction array
        #print('\nCreate submission data...')
        #submission_data = N.array(map(poisson_cumul, N.round(predictions)))
        #solution.generate_submission_file(list_id, submission_data)
        print '\n\n\n Done!'


if __name__ == "__main__":
    #a = RandomForestClf('Data/train_2013.csv', 1000)
    #a = RandomForestClf('Data/train_2013.csv', 'all')
    #a = RandomForestClf(saved_pkl='saved_clf/train_data.pkl')
    a = RandomForestClf(clf_pkl='saved_clf/clf_md20_ne250/clf.pkl')
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
    coltofit = ['Avg_Reflectivity', 'Range_Reflectivity', 'Nval',
                'Avg_DistanceToRadar', 'Avg_RadarQualityIndex',
                'Range_RadarQualityIndex',
                'Avg_RR1', 'Range_RR1', 'Range_RR2',
                'Range_RR3', 'Avg_Zdr', 'Range_Zdr',
                'Avg_Composite', 'Range_Composite', 'Avg_HybridScan',
                'Range_HybridScan', 'Avg_Velocity', 'Range_Velocity',
                'Avg_LogWaterVolume', 'Range_LogWaterVolume',
                'Range_MassWeightedMean',
                'Avg_MassWeightedSD', 'Range_MassWeightedSD',
                'Avg_RhoHV', 'Range_RhoHV',
                ]
    # Adding hydrometeor types
    # Some are merged because they have the same meaning
    #hm_types = [0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
    hm_types = [0, 1, 7, 8, 13] # only important ones
    coltofit.extend(["hm_{}".format(i) for i in hm_types])

    #print coltofit
    #a.cv_scores(coltofit)
    #a.fitNscore(coltofit)
    testplicklename = 'saved_clf/test_data.pkl'
    #a.fitNsave(coltofit, 'saved_clf/clf_md20_ne250/clf.pkl', maxdepth=20, nestimator=250)
    a.submit(coltofit, test_pickle=testplicklename)
    #a.submit(coltofit, test_pickle=testplicklename, maxdepth=18, nestimator=200)
    #a.submit(coltofit)
    #a.prepareNsave(a.df_full, coltofit,
    #                save_name='saved_clf/train_data.pkl')
    ## prepare and save test
    #a.prepareNsave(pd.read_csv('Data/test_2014.csv'),
    #               coltofit, save_name=testplicklename, ignore_clean=True)
