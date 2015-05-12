# General imports
import matplotlib.pyplot as plt
import numpy as N
from time import time
from operator import itemgetter

# Sklearn
from sklearn.learning_curve import learning_curve

from gb_reg import GBoostReg
from score import kaggle_score


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

        ## Plotting
        fig = plt.figure()
        plt.xlabel("Training examples")
        plt.ylabel('kaggle score')
        plt.title("Learning Curves (Gradient Boosting, reg.)")
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

if __name__=='__main__':
    lrn = reg_learning('Data/train_2013.csv', 100000)
    #lrn = reg_learning('Data/train_2013.csv', 1000)
    coltofit = ['Avg_Reflectivity', 'Range_Reflectivity', 'Nval',
                'Avg_DistanceToRadar', 'Avg_RadarQualityIndex', 'Range_RadarQualityIndex',
                'Avg_RR1', 'Range_RR1','Avg_RR2', 'Range_RR2',
                'Avg_RR3', 'Range_RR3', 'Avg_Zdr', 'Range_Zdr',
                'Avg_Composite', 'Range_Composite','Avg_HybridScan', 'Range_HybridScan',
                'Avg_Velocity', 'Range_Velocity', 'Avg_LogWaterVolume', 'Range_LogWaterVolume',
                'Avg_MassWeightedMean', 'Range_MassWeightedMean',
                'Avg_MassWeightedSD', 'Range_MassWeightedSD', 'Avg_RhoHV', 'Range_RhoHV'
                ]
    hm_types = [0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
    #hm_types = [0, 1, 7, 8, 13] # only important ones
    coltofit.extend(["hm_{}".format(i) for i in hm_types])
    lrn.learn_curve(coltofit, n_jobs=2, verbose=1)
