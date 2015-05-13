# General imports
import matplotlib.pyplot as plt
import numpy as N
from time import time
from operator import itemgetter

# Sklearn
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import grid_search

from gb_reg import GBoostReg
from score import kaggle_score
import feature_lists


class reg_learning(GBoostReg):

    """Class that will contain learning functions."""

    def learn_curve(self, col2fit, **kwargs):
        """Plot learning curve."""
        verbose = kwargs.get('verbose', 0)
        nsizes = kwargs.get('nsizes', 5)
        waitNshow = kwargs.get('waitNshow', True)
        n_jobs = kwargs.get('n_jobs', 1)
        cv = kwargs.get('cv', 3)
        pre_dispatch = '2*n_jobs'

        print('nsizes={}, n_jobs={}, pre_dispatch={}\n'.format(nsizes,
                                                               n_jobs,
                                                               pre_dispatch))
        # Create a list of nsize incresing #-of-sample to train on
        train_sizes = [x / float(nsizes) for x in range(1, nsizes + 1)]
        print 'training will be performed on the following sizes'
        print train_sizes

        self.prepare_data(self.df_full, True, col2fit)
        train_values = self.df_full[col2fit].values
        target_values = self.df_full['Expected'].values

        print '\n\nlearning with njobs = {}\n...\n'.format(n_jobs)

        self.set_model(**kwargs)
        train_sizes, train_scores, test_scores =\
            learning_curve(self.rainRegressor,
                           train_values, target_values, cv=cv, verbose=verbose,
                           n_jobs=n_jobs, pre_dispatch=pre_dispatch,
                           train_sizes=train_sizes, scoring=kaggle_score)

        par_dict = self.rainRegressor.get_params()

        ## Plotting
        fig = plt.figure()
        plt.xlabel("Training examples")
        plt.ylabel('kaggle score')
        title_str = "Learning Curves, GBoosting reg.)"
        title_str += ", ne={}".format(par_dict['n_estimators'])
        title_str += " md={}".format(par_dict['max_depth'])
        plt.title(title_str)
        train_scores_mean = N.mean(train_scores, axis=1)
        train_scores_std = N.std(train_scores, axis=1)
        test_scores_mean = N.mean(test_scores, axis=1)
        test_scores_std = N.std(test_scores, axis=1)
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std,
                         alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
        plt.legend(loc="best")
        print('Learning curve finished')

        if waitNshow:
            fig.show()
            raw_input('press enter when finished...')
        return {'fig_learning': fig, 'train_scores': train_scores, 'test_scores':test_scores}

    def valid_curve(self, col2fit, **kwargs):
        """Plot the learning curve over raining data."""
        if not self.iscleaned:
            print 'Preparing the data...'
            self.prepare_data(self.df_full, True, col2fit)
        else:
            print 'data frame is already cleaned...'

        score = kaggle_score
        train_values = self.df_full[col2fit].values
        target_values = self.df_full['Expected'].values

        paramater4validation = 'max_depth'
        param_range = [8,16,20,24,30]

        print '\nValidating on {} with ranges:'.format(paramater4validation)
        print param_range

        n_jobs = 4
        print '\n\nwith n_jobs = {}\n...\n'.format(n_jobs)

        cv = 3
        #cv = 5
        print 'validating with {} cross validations...'.format(cv)
        train_scores, test_scores = validation_curve(
            GradientBoostingRegressor(), train_values, target_values,
            param_name=paramater4validation, param_range=param_range, cv=cv,
            scoring=score, verbose=1, n_jobs=n_jobs)


        ## plotting
        train_scores_mean = N.mean(train_scores, axis=1)
        train_scores_std = N.std(train_scores, axis=1)
        test_scores_mean = N.mean(test_scores, axis=1)
        test_scores_std = N.std(test_scores, axis=1)
        fig = plt.figure()
        plt.title("Validation Curve")
        plt.xlabel(paramater4validation)
        plt.ylabel(score)
        plt.plot(param_range, train_scores_mean, label="Training score", color="r")
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="r")
        plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="g")
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="g")
        plt.grid()
        plt.legend(loc='best')
        fig.show()
        raw_input('press enter when finished...')


    def grid_report(self, grid_scores, n_top=5):
        """Utility function to report best scores."""
        top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
        for i, score in enumerate(top_scores):
            print("Model with rank: {0}".format(i + 1))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                score.mean_validation_score,
                N.std(score.cv_validation_scores)))
            print("Parameters: {0}".format(score.parameters))
            print("")

    def grid_search(self, col2fit, **kwargs):
        """Using grid search to find the best parameters."""
        n_jobs = kwargs.get('n_jobs', 1)

        # use a full grid over all parameters
        parameters = {"max_depth": [3, 6, 24],
                      "max_features": [1.0, 0.3, 0.1],
                      "min_samples_leaf": [3, 5, 9, 17],
                      "learning_rate": [0.01, 0.02, 0.05, 0.1]}

        if not self.iscleaned:
            print 'Preparing the data...'
            self.prepare_data(self.df_full, True, col2fit)
        else:
            print 'data frame is already cleaned...'
        train_values = self.df_full[col2fit].values
        target_values = self.df_full['Expected'].values

        pre_dispatch = '2*n_jobs'

        # Fit the grid
        print 'fitting the grid with njobs = {}...'.format(n_jobs)
        start = time()
        estimator = GradientBoostingRegressor(n_estimators=200)
        rf_grid = grid_search.RandomizedSearchCV(estimator,
                                                 parameters,
                                                 n_jobs=n_jobs, verbose=2,
                                                 pre_dispatch=pre_dispatch,
                                                 error_score=0,
                                                 n_iter=20)
        rf_grid.fit(train_values, target_values)
        print('Grid search finished')

        print("\n\nGridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(rf_grid.grid_scores_)))
        self.grid_report(rf_grid.grid_scores_)

        print('\n\nBest score = {}'.format(rf_grid.best_score_))
        print('Best params = {}\n\n'.format(rf_grid.best_params_))


if __name__=='__main__':
    #lrn = reg_learning('Data/train_2013.csv', nrows=None)
    #lrn = reg_learning('Data/train_2013.csv', 200000)
    lrn = reg_learning('Data/train_2013.csv', 1000)
    #lrn = reg_learning(saved_pkl='saved_clf/train_data_700k.pkl')
    #lrn = reg_learning(saved_pkl='saved_clf/train_data.pkl')
    lrn.learn_curve(feature_lists.list1, n_jobs=6, n_estimators=100)
    #lrn.grid_search(coltofit, n_jobs=4, verbose=1)
    #lrn.valid_curve(coltofit)
