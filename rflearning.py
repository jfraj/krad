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



if __name__=='__main__':
    lrn = clf_learning('Data/train_2013.csv', 200000)
    coltofit = ['Avg_Reflectivity', 'Range_Reflectivity', 'Nval',
                'Avg_DistanceToRadar', 'Avg_RadarQualityIndex', 'Range_RadarQualityIndex',
                'Avg_RR1', 'Range_RR1','Avg_RR2', 'Range_RR2',
                'Avg_RR3', 'Range_RR3',
                ]
    lrn.learn_curve(coltofit)
