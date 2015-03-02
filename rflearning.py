#================================================================================
#
# Class for creating random forest learning curves
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
from sklearn import grid_search

from rf2steps import RandomForestModel



class clf_learning(RandomForestModel):
    """
    Class that will contain learning
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
        njobs = max(1, int(multiprocessing.cpu_count()/2))
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

    def grid_search(self, col2fit):
        """
        Using grid search to find the best parameters
        """
        #max_depths = [2,3,4,5,6,7,8,9,11,15,20]
        #nestimators = [5, 10, 20, 30, 50, 70, 80, 100, 150, 200]
        max_depths = [8,12,16,20,24]
        nestimators = [100, 150, 200, 250, 300, 350, 400]
        parameters = {'max_depth': max_depths, 'n_estimators' : nestimators}

        self.prepare_data(self.df_full, True, col2fit)
        train_values = self.df_full[col2fit].values
        target_values = self.df_full['rain'].values

        ## Number of cpu to use
        ## Making sure there is one free unless there is only one
        #njobs = max(1, int(0.75*multiprocessing.cpu_count()))
        njobs = max(1, int(multiprocessing.cpu_count() -1))
        
        ## Fit the grid
        print 'fitting the grid with njobs = {}...'.format(njobs)
        rf_grid = grid_search.GridSearchCV(RandomForestClassifier(), parameters)
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
        print '----------------------'
        print 'best parameters:'
        print rf_grid.best_params_
        print 
        print rf_grid.best_score_
        raw_input('press enter to finished...')



if __name__=='__main__':
    lrn = clf_learning('Data/train_2013.csv', 200000)
    coltofit = ['Avg_Reflectivity', 'Range_Reflectivity', 'Nval',
                'Avg_DistanceToRadar', 'Avg_RadarQualityIndex', 'Range_RadarQualityIndex',
                'Avg_RR1', 'Range_RR1','Avg_RR2', 'Range_RR2',
                'Avg_RR3', 'Range_RR3',
                ]
    #lrn.learn_curve(coltofit)
    lrn.grid_search(coltofit)
