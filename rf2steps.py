## General imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as N

## Sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.learning_curve import validation_curve
from sklearn.learning_curve import learning_curve
from sklearn import cross_validation

## Ressources
import multiprocessing
import gc

import clean, solution
from score import kaggle_metric, heaviside, poisson_cumul


class RandomForestModel(object):
    """
    A class that will contain and train data for random forest
    """
    def __init__(self, train_data_fname, nrows = 'all'):
        """
        Turn data in pandas dataframe
        """
        if nrows == 'all':
            self.df_full = pd.read_csv(train_data_fname)
        else:
            self.df_full = pd.read_csv(train_data_fname, nrows=nrows)
        print 'Creating training data frame with shape'
        print self.df_full.shape

        ##Define the classifier and regressor variables
        self.rainClassifier = None
        self.rainRegressor = None

    def prepare_data(self, df, verbose = False, var2prep = 'all'):
        """
        prepare self.df_full for fitting
        var2prep is a list of variables that will be needed.
        This will save time by cleaning only the needed variables
        """
        if verbose:
            print 'Getting radar length'
        df['RadarCounts'] = df['TimeToEnd'].apply(clean.getRadarLength)

        ## Add a category column rain/norain (1/0)
        ## Might consider using a threshold i.e. rain if Expected > threshold
        if 'Expected' in df.columns.values:
            df['rain'] = df['Expected'].apply(lambda x: 1 if x>0 else 0)
        
        #######
        ## Adding the mean of variables to fit

        ## Reflectivity
        if var2prep == 'all' or any("Reflectivity" in s for s in var2prep):
            if verbose:
                print 'Clean reflectivity'
            df['Reflectivity1'] = df[['RadarCounts','Reflectivity']].apply(
                clean.getIthRadar, axis=1)
            df['Avg_Reflectivity'],  df['Range_Reflectivity'], df['Nval']=\
              zip(*df['Reflectivity1'].apply(clean.getListReductions))

        ##Zdr
        if var2prep == 'all' or any("Zdr" in s for s in var2prep):
            if verbose:
                print 'Clean Zdr'
            df['Zdr1'] = df[['RadarCounts','Zdr']].apply(clean.getIthRadar, axis=1)
            df['Avg_Zdr'],  df['Range_Zdr'], df['Nval_Zdr']=\
              zip(*df['Zdr1'].apply(clean.getListReductions))
            df.drop('Nval_Zdr', axis=1, inplace=True)# Already in Nval
            ## Set negative RR1 (could not be computed) to 0.0 i.e. no rain
            ## (elements in the list with error code (<=-99000) will make the average negative)
            df.loc[df.Avg_Zdr < 0, 'Avg_Zdr'] = 0.0
            df.loc[df.Range_Zdr > 1000, 'Range_Zdr'] = 0.0

        ##Composite
        if var2prep == 'all' or any("Composite" in s for s in var2prep):
            if verbose:
                print 'Clean Composite'
            df['Composite1'] = df[['RadarCounts','Composite']].apply(clean.getIthRadar, axis=1)
            df['Avg_Composite'],  df['Range_Composite'], df['Nval_Composite']=\
              zip(*df['Composite1'].apply(clean.getListReductions))
            df.drop('Nval_Composite', axis=1, inplace=True)# Already in Nval
            ## Set negative Composite (could not be computed) to 0.0 i.e. no rain
            ## (elements in the list with error code (<=-99000) will make the average negative)
            df.loc[df.Avg_Composite < 0, 'Avg_Composite'] = 0.0
            df.loc[df.Range_Composite > 1000, 'Range_Composite'] = 0.0

        ##HybridScan
        if var2prep == 'all' or any("HybridScan" in s for s in var2prep):
            if verbose:
                print 'Clean HybridScan'
            df['HybridScan1'] = df[['RadarCounts','HybridScan']].apply(clean.getIthRadar, axis=1)
            df['Avg_HybridScan'],  df['Range_HybridScan'], df['Nval_HybridScan']=\
              zip(*df['HybridScan1'].apply(clean.getListReductions))
            df.drop('Nval_HybridScan', axis=1, inplace=True)# Already in Nval
            ## Set negative HybridScan (could not be computed) to 0.0 i.e. no rain
            ## (elements in the list with error code (<=-99000) will make the average negative)
            df.loc[df.Avg_HybridScan < 0, 'Avg_HybridScan'] = 0.0
            df.loc[df.Range_HybridScan > 1000, 'Range_HybridScan'] = 0.0

        ##Velocity
        if var2prep == 'all' or any("Velocity" in s for s in var2prep):
            if verbose:
                print 'Clean Velocity'
            df['Velocity1'] = df[['RadarCounts','Velocity']].apply(clean.getIthRadar, axis=1)
            df['Avg_Velocity'],  df['Range_Velocity'], df['Nval_Velocity']=\
              zip(*df['Velocity1'].apply(clean.getListReductions))
            df.drop('Nval_Velocity', axis=1, inplace=True)# Already in Nval
            ## Set negative Velocity (could not be computed) to 0.0 i.e. no rain
            ## (elements in the list with error code (<=-99000) will make the average negative)
            df.loc[df.Avg_Velocity < 0, 'Avg_Velocity'] = 0.0
            df.loc[df.Range_Velocity > 1000, 'Range_Velocity'] = 0.0

        ##LogWaterVolume
        if var2prep == 'all' or any("LogWaterVolume" in s for s in var2prep):
            if verbose:
                print 'Clean LogWaterVolume'
            df['LogWaterVolume1'] = df[['RadarCounts','LogWaterVolume']].apply(clean.getIthRadar, axis=1)
            df['Avg_LogWaterVolume'],  df['Range_LogWaterVolume'], df['Nval_LogWaterVolume']=\
              zip(*df['LogWaterVolume1'].apply(clean.getListReductions))
            df.drop('Nval_LogWaterVolume', axis=1, inplace=True)# Already in Nval
            df['Avg_LogWaterVolume'].fillna(0, inplace=True)
            df['Range_LogWaterVolume'].fillna(0, inplace=True)



        ## Distance to radar
        if var2prep == 'all' or any("DistanceToRadar" in s for s in var2prep):
            if verbose:
                print 'Clean DistanceToRadar'
            df['DistanceToRadar1'] = df[['RadarCounts','DistanceToRadar']].apply(clean.getIthRadar, axis=1)
            df['Avg_DistanceToRadar'],  df['Range_DistanceToRadar'], df['Nval_DistanceToRadar']=\
              zip(*df['DistanceToRadar1'].apply(clean.getListReductions))
            ## Remove the Nval_xxx it's already in the Nval column 
            df.drop('Nval_DistanceToRadar', axis=1, inplace=True)

        ## Radar quality index
        if var2prep == 'all' or any("RadarQualityIndex" in s for s in var2prep):
            if verbose:
                print 'Clean RadarQualityIndex'
            df['RadarQualityIndex1'] =\
              df[['RadarCounts','RadarQualityIndex']].apply(clean.getIthRadar,axis=1)
            df['Avg_RadarQualityIndex'],  df['Range_RadarQualityIndex'], df['Nval_RadarQualityIndex']=\
              zip(*df['RadarQualityIndex1'].apply(clean.getListReductions))
            df.drop('Nval_RadarQualityIndex', axis=1, inplace=True)# Already in Nval
            ## Set Avg_RadarQualityIndex above 1 (could not be computed) to 0.5 i.e. average data
            ## (any element in the list above 999 will make the average above 1)
            df.loc[df.Avg_RadarQualityIndex > 1, 'Avg_RadarQualityIndex'] = 0.5
            ## Set All the < 0 (something wrong with measurement) as 0 i.e. bad data
            df.loc[df.Avg_RadarQualityIndex < 0, 'Avg_RadarQualityIndex'] = 0.0

        ##RR1
        if var2prep == 'all' or any("RR1" in s for s in var2prep):
            if verbose:
                print 'Clean RR1'
            df['RR11'] = df[['RadarCounts','RR1']].apply(clean.getIthRadar, axis=1)
            df['Avg_RR1'],  df['Range_RR1'], df['Nval_RR1']=\
              zip(*df['RR11'].apply(clean.getListReductions))
            df.drop('Nval_RR1', axis=1, inplace=True)# Already in Nval
            ## Set negative RR1 (could not be computed) to 0.0 i.e. no rain
            ## (elements in the list with error code (<=-99000) will make the average negative)
            df.loc[df.Avg_RR1 < 1, 'Avg_RR1'] = 0.0

        ##RR2
        if var2prep == 'all' or any("RR2" in s for s in var2prep):
            if verbose:
                print 'Clean RR2'
            df['RR21'] = df[['RadarCounts','RR2']].apply(clean.getIthRadar, axis=1)
            df['Avg_RR2'],  df['Range_RR2'], df['Nval_RR2']=\
              zip(*df['RR21'].apply(clean.getListReductions))
            df.drop('Nval_RR2', axis=1, inplace=True)# Already in Nval
            ## Set negative RR2 (could not be computed) to 0.0 i.e. no rain
            ## (elements in the list with error code (<=-99000) will make the average negative)
            df.loc[df.Avg_RR2 < 1, 'Avg_RR2'] = 0.0

        ##RR3
        if var2prep == 'all' or any("RR3" in s for s in var2prep):
            if verbose:
                print 'Clean RR3'
            df['RR31'] = df[['RadarCounts','RR3']].apply(clean.getIthRadar, axis=1)
            df['Avg_RR3'],  df['Range_RR3'], df['Nval_RR3']=\
              zip(*df['RR31'].apply(clean.getListReductions))
            df.drop('Nval_RR3', axis=1, inplace=True)# Already in Nval
            ## Set negative RR3 (could not be computed) to 0.0 i.e. no rain
            ## (elements in the list with error code (<=-99000) will make the average negative)
            df.loc[df.Avg_RR3 < 1, 'Avg_RR3'] = 0.0
        ##

    def fitClassifier(self, col2fit, maxdepth = 8, nestimators = 40, nrows = 'all'):
        """
        Fit the classifier for rain/norain
        """        
        ##Fit whether it rained or not with a classifier
        print '\nFitting classifier for rain-norain with max_depth={} and n_estimators={} the following columns:'.format(maxdepth, nestimators)
        print col2fit
        print 'Using {} rows'.format(nrows)
        if nrows == 'all':
            nrows = self.df_full.shape[0]
        print 'nrows = %d'%nrows
        values2fit = self.df_full[:nrows][col2fit].values
        targets = self.df_full[:nrows]['rain'].values
        self.rainClassifier = RandomForestClassifier(n_estimators=nestimators, max_depth=maxdepth)
        
        print '\nFitting...'
        self.rainClassifier.fit(values2fit, targets)

        print 'Done!\n\nFeatures importances'
        ord_idx = N.argsort(self.rainClassifier.feature_importances_)#Feature index ordered by importance 
        for ifeaturindex in ord_idx[::-1]:
            print '{0} \t: {1}'.format(col2fit[ifeaturindex], round(self.rainClassifier.feature_importances_[ifeaturindex], 2))
        print("Classifier (self) score is ", self.rainClassifier.score(values2fit, targets))

    def fitRegressor(self, col2fit, maxdepth = 8, nestimators = 40,nrows = 'all'):
        """
        Fit the regressor for the amount of rain
        """
        if nrows == 'all':
            nrows = self.df_full.shape[0]
        print '\nFitting Regressor only raining data with max_depth={} and n_estimators={} the following columns:'.format(maxdepth, nestimators)
        values2fit = self.df_full[:nrows][self.df_full[:nrows]['rain'] == 1][['Expected'] + col2fit].values
        self.rainRegressor = RandomForestRegressor(n_estimators=nestimators, max_depth=maxdepth)

        print '\nFitting on the {} rain samples...'.format(values2fit.shape[0])
        self.rainRegressor.fit(values2fit[:,1:], values2fit[:,0])

        print 'Done!\n\nFeatures importances'
        ord_idx = N.argsort(self.rainRegressor.feature_importances_)#Feature index ordered by importance 
        for ifeaturindex in ord_idx[::-1]:
            print '{0} \t: {1}'.format(col2fit[ifeaturindex], round(self.rainRegressor.feature_importances_[ifeaturindex], 2))
            
    def fitNscoreClassifier(self, col2fit, maxdepth=8, nestimators=40):
        """
        Fit on one fraction of the data and score on the rest
        """

        print 'Preparing the data...'
        self.prepare_data(self.df_full, True, col2fit)

        ## number of rows used for the fit
        nrows = self.df_full.shape[0]
        nfit = int(0.7*nrows)
        nscore = nrows - nfit
        print '\nTraining will be performed with {} rows and scored on {} rows\n'.format(nfit, nscore)


        ## Fit rain-norain
        rfmodel.fitClassifier(col2fit, maxdepth, nestimators, nfit)

        ## Cross validate on independant samples
        values2val = self.df_full[nfit:][col2fit].values
        target2val = self.df_full[nfit:]['rain'].values

        print 'Cross validating on {} rows'.format(values2val.shape[0])
        
        scores = cross_validation.cross_val_score(self.rainClassifier, values2val, target2val, cv=10)
        print scores
        print '\n\nCross validation accuracy: %.2f (+/- %.3f)\n' % (round(scores.mean(), 2), round(scores.std() / 2, 3))

    def fitNscoreRegressor(self, col2fit, maxdepth=8, nestimators=40):
        """
        Fit the regressor only on the data with rain
        """
        print 'Preparing the data...'
        self.prepare_data(self.df_full, True, col2fit)

        ## number of rows used for the fit
        nrows = self.df_full.shape[0]
        nfit = int(0.7*nrows)## The fit will be performed on the [:nfit] rows where expected > 0

        ## Fit only where it rained
        rfmodel.fitRegressor(col2fit, maxdepth, nestimators, nfit)

        ## Cross validate on independant samples
        values2val = self.df_full[nfit:][self.df_full[nfit:]['rain'] == 1][col2fit].values
        target2val = self.df_full[nfit:][self.df_full[nfit:]['rain'] == 1]['Expected'].values

        print 'Cross validating on {} rows'.format(values2val.shape[0])

        ## Predict on the rest of the sample
        print '\nPredicting...'
        output = self.rainRegressor.predict(values2val)


        ## Get and print the score
        print '\nScoring (independently of classifier)...'
        score = kaggle_metric(N.round(output), target2val)
        score_pois = kaggle_metric(N.round(output), target2val, 'poisson')
        print '\n\nScore(heaviside)={}'.format(score)
        print '\nScore(poisson)={}\n\n'.format(score_pois)


    def fitNscoreAll(self, clf_col2fit, reg_col2fit):
        """
        Fit the classifier and regressor
        Calculate the score of using both
        Note: Eventually there could/should be different column to fit for the classifier and Regressor 
        """
        ##Fit parameters
        clf_maxdepth, clf_nestimators = 15, 150
        reg_maxdepth, reg_nestimators = 12, 150
        
        print 'Preparing the data...'
        combined_col = clf_col2fit + list(set(reg_col2fit) - set(clf_col2fit))
        self.prepare_data(self.df_full, True, combined_col)

        ## number of rows used for the fit
        nrows = self.df_full.shape[0]
        nfit = int(0.7*nrows)

        print 'Fitting classifier for rain/norain with maxdepth={} and nestimators={}...'.format(clf_maxdepth, clf_nestimators)
        rfmodel.fitClassifier(clf_col2fit, clf_maxdepth, clf_nestimators, nfit)

        print 'Fit regressor only where it rained to predict amount of rain with maxdepth={} and nestimators={}'.format(reg_maxdepth, reg_nestimators)
        rfmodel.fitRegressor(reg_col2fit, reg_maxdepth, reg_nestimators, nfit)

        ## Cross validate on independant samples
        clf_values2predict = self.df_full[nfit:][clf_col2fit].values

        print '\nPredicting rain/norain with classifier...'
        clf_predict = self.rainClassifier.predict(clf_values2predict)
        df_predict = self.df_full[nfit:]['Expected']

        print '\nPredicting amount of rain with regressor...'
        reg_values2predict = self.df_full[nfit:][clf_predict==1][reg_col2fit].values
        reg_predict = self.rainRegressor.predict(reg_values2predict)

        ## Creating array to compare with expected
        ## First those that were predicted as no-rain
        targets = self.df_full[nfit:][clf_predict==0]['Expected'].values
        fullpredict = N.zeros(len(self.df_full[nfit:][clf_predict==0]))
        ## Then add the rain prediction
        fullpredict = N.append(fullpredict, reg_predict)
        targets = N.append(targets, self.df_full[nfit:][clf_predict==1]['Expected'].values)
        #print zip(fullpredict, targets)
        print '\nScoring...'
        score = kaggle_metric(N.round(fullpredict), targets)
        score_pois = kaggle_metric(N.round(fullpredict), targets, 'poisson')
        print '\n\nScore(heaviside)={}'.format(score)
        print '\nScore(poisson)={}\n\n'.format(score_pois)

    def submit(self, clf_col2fit, reg_col2fit):
        """
        Create csv file for submission
        """
        ##Fit parameters
        clf_maxdepth, clf_nestimators = 15, 200
        reg_maxdepth, reg_nestimators = 12, 200

        print 'Preparing the data...'
        combined_col = clf_col2fit + list(set(reg_col2fit) - set(clf_col2fit))
        self.prepare_data(self.df_full, True, combined_col)

        rfmodel.fitClassifier(clf_col2fit, clf_maxdepth, clf_nestimators)

        print 'Fit regressor only where it rained to predict amount of rain with maxdepth={} and nestimators={}'.format(reg_maxdepth, reg_nestimators)
        rfmodel.fitRegressor(reg_col2fit, reg_maxdepth, reg_nestimators)

        print '\nGetting and cleaning all test data...'
        df_test = pd.read_csv('Data/test_2014.csv')
        #df_test = pd.read_csv('Data/test_2014.csv', nrows=2000)## For testing
        
        list_id = df_test['Id'].values
        self.prepare_data(df_test, True, combined_col)

        ## Cross validate on independant samples
        clf_values2predict = df_test[clf_col2fit].values

        print '\nPredicting rain/norain with classifier...'
        clf_predict = self.rainClassifier.predict(clf_values2predict)

        print '\nPredicting amount of rain with regressor...'
        reg_values2predict = df_test[clf_predict==1][reg_col2fit].values
        reg_predict = self.rainRegressor.predict(reg_values2predict)

        ## Creating prediction array
        ## First those that were predicted as no-rain
        fullpredict = N.zeros(len(df_test[clf_predict==0]))
        ## Then add the rain prediction
        fullpredict = N.append(fullpredict, reg_predict)

        print '\nCreate submission data...'
        submission_data = N.array(map(poisson_cumul, N.round(fullpredict)))
        ## The id have to be reorganized
        list_id = df_test[clf_predict==0]['Id'].values
        list_id = N.append(list_id, df_test[clf_predict==1]['Id'].values)
        solution.generate_submission_file(list_id,submission_data)
        print '\n\n\n Done!'


if __name__=='__main__':
    rfmodel = RandomForestModel('Data/train_2013.csv', 700000)
    #rfmodel = RandomForestModel('Data/train_2013.csv', 'all')
    #coltofit = ['Avg_Reflectivity', 'Range_Reflectivity', 'Nval', 'Avg_RR1', 'Range_RR1', 'Avg_RR2', 'Range_RR2']
    coltofit = ['Avg_Reflectivity', 'Range_Reflectivity', 'Nval',
                'Avg_DistanceToRadar', 'Avg_RadarQualityIndex', 'Range_RadarQualityIndex',
                'Avg_RR1', 'Range_RR1','Avg_RR2', 'Range_RR2',
                'Avg_RR3', 'Range_RR3', 'Avg_Zdr', 'Range_Zdr',
                'Avg_Composite', 'Range_Composite','Avg_HybridScan', 'Range_HybridScan',
                'Avg_Velocity', 'Range_Velocity', 'Avg_LogWaterVolume', 'Range_LogWaterVolume',
                ]
    clf_coltofit = coltofit
    reg_coltofit = coltofit
    #clf_coltofit = ['Avg_Reflectivity', 'Range_Reflectivity', 'Nval',
    #            'Avg_DistanceToRadar', 'Avg_RadarQualityIndex', 'Range_RadarQualityIndex',
    #            'Avg_RR1', 'Range_RR1', 'Range_RR2', 'Range_RR3',
    #            ]
    #reg_coltofit = ['Avg_Reflectivity', 'Range_Reflectivity', 'Nval',
    #            'Avg_DistanceToRadar', 'Avg_RadarQualityIndex', 'Range_RadarQualityIndex',
    #            'Avg_RR1', 'Range_RR1','Avg_RR2', 'Range_RR2',
    #            'Avg_RR3', 'Range_RR3',
    #            ]
    #reg_coltofit = ['Avg_Reflectivity', 'Range_Reflectivity', 'Nval',
    #            'Avg_DistanceToRadar', 'Avg_RadarQualityIndex', 'Range_RadarQualityIndex',
    #            'Range_RR1',
    #            ]
    rfmodel.fitNscoreAll(clf_coltofit, reg_coltofit)
    #rfmodel.submit(clf_coltofit, reg_coltofit)
