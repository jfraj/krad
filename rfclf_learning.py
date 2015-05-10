# General imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as N

## Ressources
import multiprocessing
import gc

## Sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
from sklearn import grid_search

from rfclf import RandomForestClf
from score import kaggle_score

class clf_learning(RandomForestClf):
    """Class that will contain learning functions."""
    def learn_curve(self, col2fit, **kwargs):
        """Plot learning curve."""
        verbose = kwargs.get('verbose', 0)
        nsizes = kwargs.get('nsizes', 5)
        waitNshow = kwargs.get('waitNshow', True)
        njobs = kwargs.get('njobs', 1)
        nestimators = kwargs.get('nestimators', 250)
        maxdepth = kwargs.get('maxdepth', 25)
        cv = kwargs.get('cv', 3)
        predispatch = '2*n_jobs'

        print('\nusing nestim={} & maxdepth={}'.format(nestimators, maxdepth))
        print('nsizes={}, n_jobs={}, pre_dispatch={}\n'.format(nsizes,
                                                               njobs,
                                                               predispatch))
        # Create a list of nsize incresing #-of-sample to train on
        train_sizes = [x / float(nsizes) for x in range(1, nsizes + 1)]
        print 'training will be performed on the following sizes'
        print train_sizes

        self.prepare_data(self.df_full, True, col2fit)
        train_values = self.df_full[col2fit].values
        target_values = self.df_full['rain'].values

        print '\n\nlearning with njobs = {}\n...\n'.format(njobs)
        train_sizes, train_scores, test_scores =\
            learning_curve(RandomForestClassifier(n_estimators=nestimators,
                                                  max_depth=maxdepth,
                                                  verbose=verbose,
                                                  class_weight='auto'),
                           train_values, target_values, cv=cv, verbose=verbose,
                           n_jobs=njobs, pre_dispatch=predispatch,
                           train_sizes=train_sizes,
                           scoring=kaggle_score)

        ## Plotting
        fig = plt.figure()
        plt.xlabel("Training examples")
        plt.ylabel('kaggle score')
        plt.title("Learning Curves (RandomForest n_estimators={0}, max_depth={1})".format(nestimators, maxdepth))
        train_scores_mean = N.mean(train_scores, axis=1)
        train_scores_std = N.std(train_scores, axis=1)
        test_scores_mean = N.mean(test_scores, axis=1)
        test_scores_std = N.std(test_scores, axis=1)
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
        plt.legend(loc="best")
        print 'Learning curve finished'
        if waitNshow:
            fig.show()
            raw_input('press enter when finished...')
        return {'fig_learning': fig, 'train_scores': train_scores, 'test_scores':test_scores}

    def grid_search(self, col2fit, **kwargs):
        """
        Using grid search to find the best parameters

        Kwargs:
         showNwaite (bool): show the plots and waits for the user to press enter when finished
         default:True
        """
        max_depths = [9, 12, 13, 15, 18, 22, 26]
        nestimators = [30, 50, 70, 80, 100, 150, 200, 250, 300]
        #max_depths = [8,20,30]
        #nestimators = [50, 200, 300]
        #nestimators = [10, 20, 30]
        if kwargs.has_key('max_depths'):
            max_depths = kwargs['max_depths']
        if kwargs.has_key('nestimators'):
            nestimators = kwargs['nestimators']


        parameters = {'max_depth': max_depths, 'n_estimators' : nestimators}

        if not self.iscleaned:
            print 'Preparing the data...'
            self.prepare_data(self.df_full, True, col2fit)
        else:
            print 'data frame is already cleaned...'
        train_values = self.df_full[col2fit].values
        target_values = self.df_full['rain'].values

        ## Number of cpu to use
        ## Making sure there is one free unless there is only one
        #njobs = max(1, int(0.75*multiprocessing.cpu_count()))
        #njobs = max(1, int(multiprocessing.cpu_count() -1))
        njobs = 1
        pre_dispatch = 1

        ## Fit the grid
        print 'fitting the grid with njobs = {}...'.format(njobs)
        rf_grid = grid_search.RandomizedSearchCV(RandomForestClassifier(class_weight='auto'),
                                                 parameters,
                                                 n_jobs=njobs, verbose=2,
                                                 scoring=kaggle_score,
                                                 pre_dispatch=pre_dispatch,
                                                 error_score=0,
                                                 n_iter=17)
        #rf_grid = grid_search.GridSearchCV(RandomForestClassifier(), parameters,
        #                                   n_jobs=njobs, verbose=2)
        rf_grid.fit(train_values, target_values)
        print('Grid search finished')

        ## Get score
        score_dict = rf_grid.grid_scores_
        nestims = []
        mxdeps = []
        scors = []
        score_factor = -50000
        for iscore in score_dict:
            print iscore[0]
            print iscore[1]
            print iscore[2]
            scors.append(iscore[1]*score_factor)
            nestims.append(iscore[0]['n_estimators'])
            mxdeps.append(iscore[0]['max_depth'])
            print(iscore)
        print('\n\n {}'.format(scors))
        fig_nestim = plt.figure()
        plt.scatter(nestims, mxdeps, s=scors)
        plt.scatter(rf_grid.best_params_['n_estimators'],
                    rf_grid.best_params_['max_depth'], c='r',
                    s=score_factor*rf_grid.best_score_)
        fig_nestim.show()

        print('\n\nBest score = {}'.format(rf_grid.best_score_))
        print('Best params = {}\n\n'.format(rf_grid.best_params_))

        raw_input('\n press enter when finished...')

if __name__=='__main__':
    #lrn = clf_learning('Data/train_2013.csv', 10000)
    lrn = clf_learning(saved_pkl='saved_clf/train_data_700k.pkl')
    #lrn = clf_learning(saved_pkl='saved_clf/train_data.pkl')
    clf_coltofit = ['Avg_Reflectivity', 'Range_Reflectivity', 'Nval',
                'Avg_DistanceToRadar', 'Avg_RadarQualityIndex', 'Range_RadarQualityIndex',
                'Avg_RR1', 'Range_RR1','Avg_RR2', 'Range_RR2',
                'Avg_RR3', 'Range_RR3', 'Avg_Zdr', 'Range_Zdr',
                'Avg_Composite', 'Range_Composite','Avg_HybridScan', 'Range_HybridScan',
                'Avg_Velocity', 'Range_Velocity', 'Avg_LogWaterVolume', 'Range_LogWaterVolume',
                'Avg_MassWeightedMean', 'Range_MassWeightedMean',
                'Avg_MassWeightedSD', 'Range_MassWeightedSD', 'Avg_RhoHV', 'Range_RhoHV'
                ]
    #lrn.grid_search(clf_coltofit)
    lrn.learn_curve(clf_coltofit, njobs=1, verbose=1,
                    nestimators=200, maxdepth=26)
