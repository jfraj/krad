#================================================================================
#
# Class for creating random forest regressor learning/validation curves and grid search 
#
#================================================================================

## General imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as N

## Ressources
import multiprocessing
import gc

## Sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
from sklearn import grid_search

from rf2steps import RandomForestModel


class reg_learning(RandomForestModel):
    """
    Class that will contain learning
    """
    def learn_curve(self, col2fit, score='r2', maxdepth=8, nestimators=40, verbose=0):
        """
        Plots the learning curve over raining data
        """
        self.prepare_data(self.df_full, True, col2fit)

        print 'Out of {} rows...'.format(self.df_full.shape[0])

        ##Drop the rows where it did not rain
        self.df_full = self.df_full[self.df_full['rain'] >0]
        print '...{} have rain and will be used for training'.format(self.df_full.shape[0])

        train_values = self.df_full[col2fit].values
        target_values = self.df_full['Expected'].values

        ##Create a list of nsize incresing #-of-sample to train on
        nsizes = 5
        train_sizes = [x / float(nsizes) for x in range(1, nsizes + 1)]

        ## Number of cpu to use
        ## Making sure there is one free unless there is only one
        #njobs = max(1, int(multiprocessing.cpu_count()/2))
        #njobs = max(1, int(0.75*multiprocessing.cpu_count()))
        njobs = max(1, multiprocessing.cpu_count()-1)
        print '\n\nlearning with njobs = {}\n...\n'.format(njobs)

        train_sizes, train_scores, test_scores = learning_curve(RandomForestRegressor(n_estimators=nestimators, max_depth=maxdepth, verbose=verbose), train_values, target_values, cv=10, n_jobs=njobs, train_sizes=train_sizes, scoring=score)

        ## Plotting
        fig = plt.figure()
        plt.xlabel("Training examples")
        plt.ylabel(score)
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
        print 'Done'
        fig.show()
        #plt.savefig('learningcurve.png')

        #print train_scores_mean
        raw_input('press enter when finished...')

        
    def valid_curve(self, col2fit, score='r2', verbose=0):
        """
        Plots the learning curve over raining data
        """
        self.prepare_data(self.df_full, True, col2fit)

        print 'Out of {} rows...'.format(self.df_full.shape[0])

        ##Drop the rows where it did not rain
        self.df_full = self.df_full[self.df_full['rain'] >0]
        print '...{} have rain and will be used for training'.format(self.df_full.shape[0])

        train_values = self.df_full[col2fit].values
        target_values = self.df_full['Expected'].values

        #paramater4validation = "max_depth"
        #nestimators = 150
        #param_range = [8, 10, 12, 14, 15, 16, 17, 18, 20, 24]
        paramater4validation = "n_estimators"
        maxdepth = 16
        #param_range = [10, 50, 100, 150, 200, 250, 300, 400, 600, 1000]
        param_range = [40, 100, 200, 300, 500, 800, 1000]

        print '\nValidating on {} with ranges:'.format(paramater4validation)
        print param_range

        njobs = max(2, multiprocessing.cpu_count()-1)
        print '\n\nUsing with njobs = {}\n...\n'.format(njobs)

        #ncrossval = 10
        ncrossval = 5
        print 'validating with {} cross validations...'.format(ncrossval)
        train_scores, test_scores = validation_curve(
            RandomForestRegressor(max_depth = maxdepth), train_values, target_values,
            param_name=paramater4validation, param_range=param_range,cv=ncrossval,
            scoring=score, verbose = verbose, n_jobs=njobs)


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

    def grid_search(self, col2fit, score='r2'):
        """
        Using grid search to find the best parameters
        """
        #max_depths = [6,8,16,24,32,40,50,60]
        #nestimators = [20, 50, 100, 150, 200, 250, 300, 400, 500]
        max_depths = [10,11,12,13,14,15]
        nestimators = [30, 50, 100, 150,200,250,300, 400, 600,800,1000]
        parameters = {'max_depth': max_depths, 'n_estimators' : nestimators}

        self.prepare_data(self.df_full, True, col2fit)

        print 'Out of {} rows...'.format(self.df_full.shape[0])

        ##Drop the rows where it did not rain
        self.df_full = self.df_full[self.df_full['rain'] >0]
        print '...{} have rain and will be used for training'.format(self.df_full.shape[0])

        train_values = self.df_full[col2fit].values
        target_values = self.df_full['Expected'].values

        njobs = max(1, int(multiprocessing.cpu_count() -1))

        ## Fit the grid
        print 'fitting the grid with njobs = {}...'.format(njobs)
        rf_grid = grid_search.GridSearchCV(RandomForestRegressor(),
                                           parameters, scoring=score, verbose=2,n_jobs=njobs, cv=5)
        rf_grid.fit(train_values, target_values)

        ## Get score
        score_dict = rf_grid.grid_scores_
        scores = [x[1] for x in score_dict]
        scores = N.array(scores).reshape(len(max_depths), len(nestimators))

        ## Plot
        fig = plt.figure()
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.spectral)
        plt.colorbar()
        plt.ylabel('max_depths')
        plt.yticks(N.arange(len(max_depths)), max_depths)
        plt.xlabel('n_estimators')
        plt.xticks(N.arange(len(nestimators)), nestimators)
        plt.gca().invert_yaxis()
        fig.show()
        print '\n\n----------------------'
        my_score_dic = {}
        print score_dict
        for iscore in score_dict:
            #print iscore[1]
            #print iscore[2]
            my_score_dic[iscore[1]] = {'params': iscore[0], 'std': iscore[2].std()}
        mean_score_list = my_score_dic.keys()
        mean_score_list.sort()
        for imeanscore in mean_score_list:
            print '{}+-{}: md={}, ne={}'.format(round(imeanscore,2),
                                                round(my_score_dic[imeanscore]['std'],3),
                                                my_score_dic[imeanscore]['params']['max_depth'],
                                                my_score_dic[imeanscore]['params']['n_estimators'])
        print '\n----------------------'
        print 'best parameters:'
        print rf_grid.best_params_
        print 
        print rf_grid.best_score_
        raw_input('press enter to finished...')

        
if __name__=='__main__':
    #lrn = reg_learning('Data/train_2013.csv', 'all')
    lrn = reg_learning('Data/train_2013.csv', 700000)
    coltofit = ['Avg_Reflectivity', 'Range_Reflectivity', 'Nval',
                'Avg_DistanceToRadar', 'Avg_RadarQualityIndex', 'Range_RadarQualityIndex',
                'Avg_RR1', 'Range_RR1','Avg_RR2', 'Range_RR2',
                'Avg_RR3', 'Range_RR3',
                ]
    #coltofit = ['Avg_Reflectivity', 'Range_Reflectivity', 'Nval',
    #            'Avg_DistanceToRadar', 'Avg_RadarQualityIndex', 'Range_RadarQualityIndex',
    #            'Range_RR1',
    #            ]
    #lrn.learn_curve(coltofit, 'r2', 12, 150)
    lrn.valid_curve(coltofit, 'r2',2)
    #lrn.grid_search(coltofit)
