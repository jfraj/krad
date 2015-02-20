#================================================================================
#
# Class for random forest training
#
#================================================================================

## General imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as N

## Sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.learning_curve import validation_curve
from sklearn.learning_curve import learning_curve

## krad import
import clean, solution
from score import kaggle_metric, heaviside, poisson_cumul

feature_dic = {\
               'Avg_Reflectivity': {'range':[-10, 50], 'log': False},
               'Expected': {'range': [-10, 70], 'log': True},
               'Avg_RR1': {'range': [-10, 70], 'log': True},
               }


class RandomForestModel(object):
    """
    A class that will contain and train data for random forest
    """
    def __init__(self, train_data_fname, nrows = 'all'):
        """
        Turn data in pandas dataframe
        """
        if nrows == 'all':
            self.df_train = pd.read_csv(train_data_fname)
        else:
            self.df_train = pd.read_csv(train_data_fname, nrows=nrows)
        print 'Creating training data frame with shape'
        print self.df_train.shape

    def prepare_data(self, df):
        """
        prepare self.df_train for fitting
        """
        df['RadarCounts'] = df['TimeToEnd'].apply(clean.getRadarLength)

        ## Adding the mean of variables to fit

        ## Reflectivity
        df['Reflectivity1'] = df[['RadarCounts','Reflectivity']].apply(
            clean.getIthRadar, axis=1)
        df['Avg_Reflectivity'],  df['Range_Reflectivity'], df['Nval']=\
          zip(*df['Reflectivity1'].apply(clean.getListReductions))

        ## Distance to radar
        df['DistanceToRadar1'] = df[['RadarCounts','DistanceToRadar']].apply(clean.getIthRadar,
                                                                             axis=1)
        df['Avg_DistanceToRadar'],  df['Range_DistanceToRadar'], df['Nval_DistanceToRadar']=\
          zip(*df['DistanceToRadar1'].apply(clean.getListReductions))
        ## Remove the Nval_xxx it's already in the Nval column 
        df.drop('Nval_DistanceToRadar', axis=1, inplace=True)

        ## Radar quality index
        df['RadarQualityIndex1'] = df[['RadarCounts','RadarQualityIndex']].apply(clean.getIthRadar,
                                                                                 axis=1)
        df['Avg_RadarQualityIndex'],  df['Range_RadarQualityIndex'], df['Nval_RadarQualityIndex']=\
          zip(*df['RadarQualityIndex1'].apply(clean.getListReductions))
        df.drop('Nval_RadarQualityIndex', axis=1, inplace=True)# Already in Nval
        ## Set Avg_RadarQualityIndex above 1 (could not be computed) to 0.5 i.e. average data
        ## (any element in the list above 999 will make the average above 1)
        df.loc[df.Avg_RadarQualityIndex > 1, 'Avg_RadarQualityIndex'] = 0.5
        ## Set All the < 0 (something wrong with measurement) as 0 i.e. bad data
        df.loc[df.Avg_RadarQualityIndex < 0, 'Avg_RadarQualityIndex'] = 0.0

        ##RR1
        df['RR11'] = df[['RadarCounts','RR1']].apply(clean.getIthRadar,
                                                                                 axis=1)
        df['Avg_RR1'],  df['Range_RR1'], df['Nval_RR1']=\
          zip(*df['RR11'].apply(clean.getListReductions))
        df.drop('Nval_RR1', axis=1, inplace=True)# Already in Nval
        ## Set negative RR1 (could not be computed) to 0.0 i.e. no rain
        ## (any element in the list with error code (<=-99000) will make the average negative)
        df.loc[df.Avg_RR1 < 1, 'Avg_RR1'] = 0.0


    def show_feature(self, feature):
        """
        Plots the given feature after preparing the data set
        """
        from matplotlib.colors import LogNorm
        self.prepare_data(self.df_train)
        ## Separate in rain & no-rain samples
        rain    = self.df_train['Expected'].apply(lambda n: n > 0 )
        no_rain = self.df_train['Expected'].apply(lambda n: n == 0 )
        rain_feature = self.df_train[rain][feature].get_values()
        norain_feature = self.df_train[no_rain][feature].get_values()
        ranges = [[min(rain_feature), max(rain_feature)],[0, 20]]
        log = False
        if feature_dic.has_key(feature):
                ranges[0] = feature_dic[feature]['range']
                log = feature_dic[feature]['log']

        ## Plot
        fig = plt.figure()
        ax = plt.subplot(1,2,1)
        #plt.hist([rain_feature,norain_feature],
        plt.hist([norain_feature,rain_feature],
               bins=100,normed=True,stacked=False,histtype='stepfilled',
                label=['No Rain','Rain'],alpha=0.75, range=ranges[0])
        if log:
            ax.set_yscale('log')
        plt.legend(loc='best')
        plt.title(feature)
        plt.subplot(1,2,2)
        #plt.scatter(self.df_train[rain][feature].get_values(), self.df_train[rain]['Expected'].get_values(), color='Red', label='Rain')
        plt.hist2d(self.df_train[rain][feature].get_values(), self.df_train[rain]['Expected'].get_values(), range=ranges, bins=100, norm=LogNorm())
        plt.xlabel(feature)
        plt.ylabel('Rain gauge')
        fig.show()
        raw_input('press enter when finished...')

    def fitNscore(self, col2fit, maxdepth=8, nestimators = 30):
        '''
        Fits the data and show the score
        '''
        assert(col2fit[0] == 'Expected')
        print 'Using the following columns:'
        print col2fit[1:]

        ## Get the data ready to fit
        self.prepare_data(self.df_train)
        values2fit = self.df_train[col2fit].values
        forest = RandomForestClassifier(n_estimators=nestimators, max_depth=maxdepth)
        nrows = self.df_train.shape[0]
        nfit = int(0.7*nrows)

        ## Fit on 70% of the score
        forest.fit(values2fit[:nfit,1:], values2fit[:nfit,0])

        ## Predict on the rest of the sample
        output = forest.predict(values2fit[nfit:,1:])

        print '\nFeatures importances'
        ord_idx = N.argsort(forest.feature_importances_)#Feature index ordered by importance 
        for ifeaturindex in ord_idx[::-1]:
            print '{0} \t: {1} '.format(col2fit[1:][ifeaturindex], round(forest.feature_importances_[ifeaturindex], 2))

        ## Get and print the score
        print 'Scoring...'
        score = kaggle_metric(N.round(output), values2fit[nfit:,0])
        score_pois = kaggle_metric(N.round(output), values2fit[nfit:,0], 'poisson')
        print '\n\nScore(heaviside)={}'.format(score)
        print '\nScore(poisson)={}\n\n'.format(score_pois)

    def validation_curves(self, col2fit):
        '''
        This is just a test for now
        Since crossvalidation does not take continuous variable
        So I multiple by 10 and turn into a int...
        '''
        ## Get the data ready to fit
        self.prepare_data(self.df_train)

        ## Turn expected into int
        self.df_train['Expected'] = self.df_train['Expected'].apply(lambda n: int(round(n)))
        
        values2fit = self.df_train[col2fit].values
        paramater4validation = "n_estimators"
        param_range = [2,3,4,5,6,7,8,9,11,15,20]
        train_scores, test_scores = validation_curve(
            RandomForestClassifier(), values2fit[0:,1:], values2fit[0:,0],
            param_name=paramater4validation, param_range=param_range,cv=10,
            scoring="accuracy", n_jobs=1)
        train_scores_mean = N.mean(train_scores, axis=1)
        train_scores_std = N.std(train_scores, axis=1)
        test_scores_mean = N.mean(test_scores, axis=1)
        test_scores_std = N.std(test_scores, axis=1)
        fig = plt.figure()
        plt.title("Validation Curve")
        plt.xlabel(paramater4validation)
        plt.ylabel("Score")
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

    def learning_curves(self, col2fit, score='accuracy', nestimators=40, maxdepth=8):
        """
        WARNING: turns the Expected into integer
        Creates a plot score vs # of training examples
        possible score:
        ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'precision', 'r2', 'recall', 'roc_auc']
        more info here:
        http://scikit-learn.org/stable/modules/learning_curve.html
        """

        ## Data clean up for training
        self.prepare_data(self.df_train)
        print 'Training on the following features:'
        print col2fit
        ## Turn expected into int
        self.df_train['Expected'] = self.df_train['Expected'].apply(lambda n: int(round(n)))

        train_data = self.df_train[col2fit].values
        X = train_data[0:,1:]
        y = train_data[0:,0]
        train_sizes = [x / 10.0 for x in range(1, 11)]##Can be other formats

        print 'learning...'
        train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(n_estimators=nestimators, max_depth=maxdepth), X, y, cv=10, n_jobs=1, train_sizes=train_sizes, scoring=score)

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
        raw_input('press enter when finished...')

    def submit(self, col2fit, maxdepth=8, nestimators = 30):
        '''
        Create the file to submit
        '''
        assert(col2fit[0] == 'Expected')
        print '\n\nPreparing sumission for the following variables'
        print col2fit
        ## Prepare train and test data
        print 'Cleaning train data...'
        self.prepare_data(self.df_train)
        print 'Getting and cleaning test data...'
        #df_test = pd.read_csv('Data/test_2014.csv', nrows=2000)##For testing
        df_test = pd.read_csv('Data/test_2014.csv')
        self.prepare_data(df_test)

        print 'Training with maxdepth={} and n_estimators={}...'.format(maxdepth, nestimators)
        values2fit = self.df_train[col2fit].values
        forest = RandomForestClassifier(n_estimators=nestimators, max_depth=maxdepth, n_jobs=-1)
        forest.fit(values2fit[:,1:], values2fit[:,0])

        print 'Predicting...'
        values4predict = df_test[col2fit[1:]].values
        prediction_output = forest.predict(values4predict)

        print 'Create submission data...'
        ## Heaviside
        #submission_data = N.array(map(heaviside, N.round(prediction_output)))
        submission_data = N.array(map(poisson_cumul, N.round(prediction_output)))

        ##The following is to compare heaviside with poisson
        '''
        for ipred in prediction_output:
            ipred = round(ipred)
            print ipred
            if ipred == 0:
                continue
            iheavy = heaviside(ipred)
            ipois = poisson_cumul(ipred)
            plt.bar(range(len(iheavy)), iheavy, alpha=0.4, color='Red')
            plt.bar(range(len(ipois)), ipois, alpha=0.4, color='Blue')
            plt.show()
            raw_input('press enter...')
        '''

        list_id = df_test['Id'].values
        solution.generate_submission_file(list_id,submission_data)
        #print '\n\n\n Done!'


                
if __name__=='__main__':
    #rfmodel = RandomForestModel('Data/train_2013.csv', 2000)
    rfmodel = RandomForestModel('Data/train_2013.csv', 'all')
    #rfmodel.show_feature('Avg_Reflectivity')
    coltofit = ['Expected', 'Avg_Reflectivity', 'Range_Reflectivity', 'Nval',
                'Avg_DistanceToRadar', 'Range_DistanceToRadar',
                'Avg_RadarQualityIndex', 'Range_RadarQualityIndex', 'Avg_RR1', 'Range_RR1']
    #rfmodel.fitNscore(coltofit)
    #rfmodel.validation_curves(coltofit)
    #rfmodel.learning_curves(coltofit)

    ##Submission
    rfmodel.submit(coltofit)
