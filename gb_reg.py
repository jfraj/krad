"""Module for fitting single random forest regressor."""
# Standard
import numpy as N
import matplotlib.pyplot as plt

# sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn import metrics

# this projects
from basemodel import BaseModel
from score import kaggle_metric


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

        self.iscleaned = True

    def set_model(self, **kwargs):
        """Set the model."""
        verbose = kwargs.get('verbose', 0)
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', 24)
        learning_rate = kwargs.get('learning_rate', 0.02)
        min_samples_leaf = kwargs.get('min_samples_leaf', 17)
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

        print('Fitting on values with shape:')
        print(values2fit.shape)
        print('\nFitting...')
        self.rainRegressor.fit(values2fit, targets)
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
        self.fitModel(features_train, target_train, **kwargs)


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


if __name__ == "__main__":
    a = GBoostReg('Data/train_2013.csv', 100000)
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
    hm_types = [0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
    coltofit.extend(["hm_{}".format(i) for i in hm_types])
    #a.prepare_data(a.df_full, True, coltofit)
    #a.set_model()
    a.fitNscore(coltofit)
