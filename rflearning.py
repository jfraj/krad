#================================================================================
#
# Class for creating random forest classifier learning/validation curves and grid search 
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
from sklearn import grid_search

from rf2steps import RandomForestModel



class clf_learning(RandomForestModel):
    """
    Class that will contain learning functions
    """
    def learn_curve(self, col2fit, score='accuracy', maxdepth=8, nestimators=40, verbose=0):
        """
        Plots the learning curve
        """
        self.prepare_data(self.df_full, True, col2fit)
        train_values = self.df_full[col2fit].values
        target_values = self.df_full['rain'].values

        ##Create a list of nsize incresing #-of-sample to train on
        nsizes = 10
        train_sizes = [x / float(nsizes) for x in range(1, nsizes + 1)]

        ## Number of cpu to use
        ## Making sure there is one free unless there is only one
        #njobs = max(1, int(multiprocessing.cpu_count()/2))
        #njobs = max(1, int(0.75*multiprocessing.cpu_count()))
        njobs = max(1, multiprocessing.cpu_count()-1)
        print '\n\nlearning with njobs = {}\n...\n'.format(njobs)

        train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(n_estimators=nestimators, max_depth=maxdepth, verbose=verbose), train_values, target_values, cv=10, n_jobs=njobs, train_sizes=train_sizes, scoring=score)

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
        raw_input('press enter when finished...')

    def valid_curve(self, col2fit, score='accuracy', verbose=0):
        """
        Plots the validation curve
        """
        self.prepare_data(self.df_full, True, col2fit)
        train_values = self.df_full[col2fit].values
        target_values = self.df_full['rain'].values

        ## Number of cpu to use
        ## Making sure there is one free unless there is only one
        njobs = max(1, int(0.75*multiprocessing.cpu_count()))
        print '\n\nValidating with njobs = {}\n...\n'.format(njobs)

        ## Parameter info is hard-coded for now, should be improved...

        paramater4validation = "n_estimators"
        maxdepth = 15
        param_range = [10, 50, 100, 150, 200, 250, 300, 400, 600, 800, 1000, 1500]

        #paramater4validation = "max_depth"
        #nestimators = 150
        #param_range = [8, 10, 12, 14, 15, 16, 17, 18, 20, 24]
        
        print '\nValidating on {} with ranges:'.format(paramater4validation)
        print param_range

        print 'validating...'
        train_scores, test_scores = validation_curve(
            RandomForestClassifier(max_depth = maxdepth), train_values, target_values,
            param_name=paramater4validation, param_range=param_range,cv=10,
            scoring=score, verbose = verbose, n_jobs=njobs)
        
        #train_scores, test_scores = validation_curve(
        #    RandomForestClassifier(n_estimators = nestimators), train_values, target_values,
        #    param_name=paramater4validation, param_range=param_range,cv=10,
        #    scoring=score, verbose = verbose, n_jobs=njobs)

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


    def grid_search(self, col2fit):
        """
        Using grid search to find the best parameters
        """
        #max_depths = [9,12,13,15,18,22,26,30,40]
        #nestimators = [30, 50, 70, 80, 100, 150, 200, 250, 300, 400, 600]
        max_depths = [8,20,30]
        nestimators = [50, 200, 300]
        parameters = {'max_depth': max_depths, 'n_estimators' : nestimators}

        if not self.iscleaned:
            print 'Preparing the data...'
            self.prepare_data(self.df_full, True, col2fit)
        train_values = self.df_full[col2fit].values
        target_values = self.df_full['rain'].values

        ## Number of cpu to use
        ## Making sure there is one free unless there is only one
        #njobs = max(1, int(0.75*multiprocessing.cpu_count()))
        njobs = max(1, int(multiprocessing.cpu_count() -1))
        
        ## Fit the grid
        print 'fitting the grid with njobs = {}...'.format(njobs)
        rf_grid = grid_search.GridSearchCV(RandomForestClassifier(), parameters,
                                           n_jobs=njobs, verbose=2)
        rf_grid.fit(train_values, target_values)
        print 'Grid search finished'

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
        print '----------------------'
        print 'best parameters:'
        print rf_grid.best_params_
        print 
        print rf_grid.best_score_
        raw_input('press enter to finished...')



if __name__=='__main__':
    #lrn = clf_learning('Data/train_2013.csv', 700000)
    lrn = clf_learning(saved_df='saved_df/test30k.h5')
    #lrn = clf_learning('Data/train_2013.csv', 'all')
    #clf_coltofit = ['Avg_Reflectivity', 'Nval',
    #            'Avg_RadarQualityIndex', 'Range_RadarQualityIndex',
    #            'Range_RR1', 'Range_RR2', 'Range_RR3']
    #clf_coltofit = ['Avg_Reflectivity', 'Range_Reflectivity', 'Nval',
    #            'Avg_DistanceToRadar', 'Avg_RadarQualityIndex', 'Range_RadarQualityIndex',
    #            'Avg_RR1', 'Range_RR1','Avg_RR2', 'Range_RR2',
    #            'Avg_RR3', 'Range_RR3', 'Avg_Zdr', 'Range_Zdr',
    #            ]
    clf_coltofit = ['Avg_Reflectivity', 'Range_Reflectivity', 'Nval',
                'Avg_DistanceToRadar', 'Avg_RadarQualityIndex', 'Range_RadarQualityIndex',
                'Avg_RR1', 'Range_RR1','Avg_RR2', 'Range_RR2',
                'Avg_RR3', 'Range_RR3', 'Avg_Zdr', 'Range_Zdr',
                'Avg_Composite', 'Range_Composite','Avg_HybridScan', 'Range_HybridScan',
                'Avg_Velocity', 'Range_Velocity', 'Avg_LogWaterVolume', 'Range_LogWaterVolume',
                'Avg_MassWeightedMean', 'Range_MassWeightedMean',
                'Avg_MassWeightedSD', 'Range_MassWeightedSD', 'Avg_RhoHV', 'Range_RhoHV'
                ]

    #lrn.learn_curve(clf_coltofit, 'accuracy', 15, 200,1)
    lrn.grid_search(clf_coltofit)
    #lrn.valid_curve(clf_coltofit, 'accuracy',2)
