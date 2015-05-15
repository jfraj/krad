"""Module for fitting single random forest regressor."""
# Standard
import pandas as pd
import numpy as N
import matplotlib.pyplot as plt

# sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn import metrics

# this projects
from basemodel import BaseModel
from score import kaggle_metric, poisson_cumul
import feature_lists
import solution


class GBoostReg(BaseModel):

    """Model using Gradient Boosting regressor."""

    def __init__(self, train_data_fname=None, nrows=None, **kwargs):
        """Initialize the data frame."""
        reg_pkl = kwargs.get('reg_pkl', False)
        if reg_pkl:
            print('\nUsing pickled regressor from {}'.format(reg_pkl))
            self.rainRegressor = joblib.load(reg_pkl)
            self.fitted = True
            self.iscleaned = False
            return
        self.rainRegressor = None
        self.fitted = False
        super(GBoostReg, self).__init__(train_data_fname, nrows, **kwargs)

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

        # Removing useless columns
        print('Removing useless columns...')
        to_keep = var2prep + ['Expected', 'Id']
        for icol in df.columns:
            if icol not in to_keep:
                df.drop(icol, axis=1, inplace=True)
        if 'Expected' in df.columns:
            self.add_weight_column(df)

        self.iscleaned = True

    def prepareNsave(self, df, col2save, **kwargs):
        """Clean and prepare data and save pickle it."""
        save_name = kwargs.get('save_name', 'saved_gbreg/default.pkl')
        print('\nWill prepare and save the following column')
        print(col2save)
        print('Preparing the data...')
        self.prepare_data(df, True, col2save, **kwargs)
        print('Pickling dataframe...')
        df.to_pickle(save_name)
        print('Done saving dataframe in {}'.format(save_name))


    def set_model(self, **kwargs):
        """Set the model."""
        verbose = kwargs.get('verbose', 0)
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', 23)
        learning_rate = kwargs.get('learning_rate', 0.01)
        min_samples_leaf = kwargs.get('min_samples_leaf', 13)
        max_features = kwargs.get('max_features', 0.1)
        random_state = kwargs.get('random_state', 42)

        self.rainRegressor = GradientBoostingRegressor(n_estimators=n_estimators,
                                                   max_depth=max_depth,
                                                   learning_rate=learning_rate,
                                                   min_samples_leaf=min_samples_leaf,
                                                   max_features=max_features,
                                                   verbose=verbose,
                                                   random_state=random_state)
        print('\n\nRegressor set with parameters:')
        par_dict = self.rainRegressor.get_params()
        for ipar in par_dict.keys():
            print('{}: {}'.format(ipar, par_dict[ipar]))
        print('\n\n')

    def fitModel(self, values2fit, targets, **kwargs):
        """Fit the Regressor."""
        if self.fitted:
            print('Already fitted...')
            return
        # Regressor
        self.set_model(**kwargs)

        # Weight if provided
        sample_weight = kwargs.get('sample_weight', None)
        if sample_weight is not None:
            print('...fitting with weight...')

        print('Fitting on values with shape:')
        print(values2fit.shape)
        print('\nFitting...')
        self.rainRegressor.fit(values2fit, targets, sample_weight=sample_weight)
        self.fitted = True
        print('Done fitting!')

    def fitNscore(self, col2fit, **kwargs):
        """Produce fit and score report"""

        # cleaning
        if not self.iscleaned:
            print 'Preparing the data...'
            self.prepare_data(self.df_full, True, col2fit)

        test_size = 0.25  # fraction kept for testing
        rnd_seed = 0  # for reproducibility

        features_train, features_test, target_train, target_test =\
            train_test_split(self.df_full[col2fit].values,
                             self.df_full['Expected'].values,
                             test_size=test_size,
                             random_state=rnd_seed)

        # Fit Regressor
        sample_weight = self.df_full['weight'].values
        self.fitModel(features_train, target_train,
                      sample_weight=None, **kwargs)


        # Feature index ordered by importance
        ord_idx = N.argsort(self.rainRegressor.feature_importances_)
        print("Feature ranking:")
        for ifeaturindex in ord_idx[::-1]:
            print('{0} \t: {1}'.format(col2fit[ifeaturindex], round(self.rainRegressor.feature_importances_[ifeaturindex], 2)))

        # Predict on the rest of the sample
        print('\nPredicting...')
        predictions = self.rainRegressor.predict(features_test)

        # Get and print the score
        print('\nScoring...')
        score = kaggle_metric(predictions, target_test)
        score_pois = kaggle_metric(predictions, target_test, 'poisson')
        print('\n\nKaggle score(heaviside)={}'.format(score))
        print('\nKaggle score(poisson)={}'.format(score_pois))
        print('\nR2 score={}'.format(metrics.r2_score(target_test, predictions)))
        print('\nMean Sqrare Error score={}\n'.format(metrics.mean_squared_error(target_test, predictions)))


        # Plots

        # Feature importances
        importances = self.rainRegressor.feature_importances_
        #std = N.std([tree.feature_importances_ for tree in self.rainRegressor.estimators_],
        #            axis=0)
        indices = N.argsort(importances)[::-1]
        ordered_names = [col2fit[i] for i in indices]

        fig_import = plt.figure(figsize=(10, 10))
        plt.title("Feature importances, reg")
        #plt.barh(range(len(indices)), importances[indices],
        #        color="b", xerr=std[indices], align="center",ecolor='r')
        plt.barh(range(len(indices)), importances[indices], color="b")
        plt.yticks(range(len(indices)), ordered_names)
        plt.ylim([-1, len(indices)])
        plt.ylim(plt.ylim()[::-1])
        plt.subplots_adjust(left=0.22)
        fig_import.show()


        # confusion matrix
        cm = metrics.confusion_matrix(target_test.astype(int), predictions.astype(int))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, N.newaxis]
        cm_normalized = N.clip(cm_normalized, 0.0, 0.5)

        fig_cm = plt.figure()
        ax_cm = fig_cm.add_subplot(1,1,1)
        im_cm = ax_cm.imshow(cm_normalized, interpolation='nearest')
        plt.title('Confusion mtx, reg')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        fig_cm.colorbar(im_cm)
        fig_cm.show()

        raw_input('press enter when finished...')

    def submit(self, col2fit, **kwargs):
        """Create csv file for submission."""
        csvname = kwargs.get('csvname', 'submission.csv')

        # Preparing training data
        if not self.iscleaned and not self.fitted:
            print('Preparing the data...')
            self.prepare_data(self.df_full, True, col2fit)

        # Fitting train data
        if not self.fitted:
            self.fitModel(self.df_full[col2fit].values,
                          self.df_full['Expected'].values, **kwargs)
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
            #df_test = pd.read_csv('Data/test_2014.csv', nrows=10050)
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
            ipredictions = self.rainRegressor.predict(ival2predict)
            isub_data = N.array(map(poisson_cumul, N.round(ipredictions)))
            print('writing isubmission data...')
            solution.generate_submission_file(ilist_id, isub_data,
                                              open_type=open_type,
                                              fname=csvname)
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
    a = GBoostReg('Data/train_2013.csv', 50000)
    #a.prepare_data(a.df_full, True, coltofit)
    #a.set_model()
    #a.prepareNsave(a.df_full, feature_lists.get_list1(), save_name='saved_gbreg/test1_666k.pkl')
    #a.fitNscore(feature_lists.get_list1(), n_estimators=200)
    #a.submit()
    a.submit(feature_lists.get_list1(), csvname='test_gb_submit.csv', n_estimators=300)
