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

class clf_learning(RandomForestClf):
    """Class that will contain learning functions."""
    def grid_search(self, col2fit, **kwargs):
        """
        Using grid search to find the best parameters

        Kwargs:
         showNwaite (bool): show the plots and waits for the user to press enter when finished
         default:True
        """
        max_depths = [9, 12, 13, 15, 18, 22, 26, 30]
        nestimators = [30, 50, 70, 80, 100, 150, 200, 250, 300, 400]
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
        fig_grid = plt.figure()
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.spectral)
        plt.colorbar()
        plt.ylabel('max_depths')
        plt.yticks(N.arange(len(max_depths)), max_depths)
        plt.xlabel('n_estimators')
        plt.xticks(N.arange(len(nestimators)), nestimators)
        plt.gca().invert_yaxis()

        print '----------------------'
        print 'best parameters:'
        print rf_grid.best_params_
        best_nestim = []
        best_nestim_mean = []
        best_nestim_std = []
        best_mdepth = []
        best_mdepth_mean = []
        best_mdepth_std = []
        print rf_grid.best_score_
        for iscore_named_tuple in rf_grid.grid_scores_:
            iscore_dic = iscore_named_tuple._asdict()
            iparams = iscore_dic['parameters']
            imean = iscore_dic['mean_validation_score']
            istd = iscore_dic['cv_validation_scores'].std()
            if iparams['n_estimators'] == rf_grid.best_params_['n_estimators']:
                best_mdepth.append(iparams['max_depth'])
                best_mdepth_mean.append(imean)
                best_mdepth_std.append(istd)
            if iparams['max_depth'] == rf_grid.best_params_['max_depth']:
                best_nestim.append(iparams['n_estimators'])
                best_nestim_mean.append(imean)
                best_nestim_std.append(istd)

        ##Turn list into arrays for better handling
        best_mdepth_mean = N.array(best_mdepth_mean)
        best_mdepth_std = N.array(best_mdepth_std)
        best_nestim_mean = N.array(best_nestim_mean)
        best_nestim_std = N.array(best_nestim_std)

        ## Plot max_depth
        fig_mdepth = plt.figure()
        plt.plot(best_mdepth, best_mdepth_mean)
        plt.fill_between(best_mdepth, best_mdepth_mean - best_mdepth_std,
                         best_mdepth_mean + best_mdepth_std, alpha=0.2, color="r")
        plt.title('Using n_estimators=%d'%rf_grid.best_params_['n_estimators'])
        plt.ylabel('accuracy')
        plt.xlabel('max_depth')

        ## Plot nestim
        fig_nestim = plt.figure()
        plt.plot(best_nestim, best_nestim_mean)
        plt.fill_between(best_nestim, best_nestim_mean - best_nestim_std,
                         best_nestim_mean + best_nestim_std, alpha=0.2, color="r")
        plt.title('Using max_depth=%d'%rf_grid.best_params_['max_depth'])
        plt.ylabel('accuracy')
        plt.xlabel('n_estimators')

        waitNshow = True
        if kwargs.has_key('waitNshow'):
            waitNshow = kwargs['waitNshow']
        if waitNshow:
            fig_grid.show()
            fig_mdepth.show()
            fig_nestim.show()
            raw_input('press enter to finished...')
        out_dic = {'fig_grid': fig_grid, 'fig_mdepth' :fig_mdepth, 'fig_nestim': fig_nestim}
        out_dic['grid_score'] = rf_grid.grid_scores_
        return out_dic

if __name__=='__main__':
    lrn = clf_learning('Data/train_2013.csv', 700000)
    clf_coltofit = ['Avg_Reflectivity', 'Range_Reflectivity', 'Nval',
                'Avg_DistanceToRadar', 'Avg_RadarQualityIndex', 'Range_RadarQualityIndex',
                'Avg_RR1', 'Range_RR1','Avg_RR2', 'Range_RR2',
                'Avg_RR3', 'Range_RR3', 'Avg_Zdr', 'Range_Zdr',
                'Avg_Composite', 'Range_Composite','Avg_HybridScan', 'Range_HybridScan',
                'Avg_Velocity', 'Range_Velocity', 'Avg_LogWaterVolume', 'Range_LogWaterVolume',
                'Avg_MassWeightedMean', 'Range_MassWeightedMean',
                'Avg_MassWeightedSD', 'Range_MassWeightedSD', 'Avg_RhoHV', 'Range_RhoHV'
                ]
    lrn.grid_search(clf_coltofit)
