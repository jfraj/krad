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
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

## Ressources
import multiprocessing
import gc

import clean, solution
from score import kaggle_metric, heaviside, poisson_cumul


class RandomForestModel(object):
    """
    A class that will contain and train data for random forest
    """

    def __init__(self, train_data_fname=None, nrows = 'all', verbose=True, **kwargs):
        """
        Turn data in pandas dataframe
        """

        ## Define the classifier and regressor variables
        self.rainClassifier = None
        self.rainRegressor = None
        self.iscleaned = False
        
        if 'saved_df' in kwargs.keys():
            print 'Getting saved training data from {}'.format(kwargs['saved_df'])
            self.df_full = pd.read_hdf(kwargs['saved_df'], 'dftest')
            print 'Training data frame has shape'
            print self.df_full.shape
            self.iscleaned = True
            return

        if train_data_fname == None:
            print 'Data will not be created by reading csv file'
            self.df_full == None
        if nrows == 'all':
            self.df_full = pd.read_csv(train_data_fname)
        else:
            self.df_full = pd.read_csv(train_data_fname, nrows=nrows)

        if verbose:
            print 'Training data frame has shape'
            print self.df_full.shape


    def prepare_data(self, df, verbose = False, var2prep = 'all'):
        """
        prepare self.df_full for fitting
        var2prep is a list of variables that will be needed.
        This will save time by cleaning only the needed variables
        """
        if verbose:
            print 'Getting radar length'
        df['RadarCounts'] = df['TimeToEnd'].apply(clean.getRadarLength)

        ## Drop rows where the expected rain is above 70
        ## This will also exclude them for the scoring (our scoring, not the kaggle one)
        if 'Expected' in df.columns.values:
            df.drop(df[df['Expected']>70].index, inplace=True)

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

        ##MassWeightedMean
        if var2prep == 'all' or any("MassWeightedMean" in s for s in var2prep):
            if verbose:
                print 'Clean MassWeightedMean'
            df['MassWeightedMean1'] = df[['RadarCounts','MassWeightedMean']].apply(clean.getIthRadar, axis=1)
            df['Avg_MassWeightedMean'],  df['Range_MassWeightedMean'], df['Nval_MassWeightedMean']=\
              zip(*df['MassWeightedMean1'].apply(clean.getListReductions))
            df.drop('Nval_MassWeightedMean', axis=1, inplace=True)# Already in Nval
            df['Avg_MassWeightedMean'].fillna(0, inplace=True)
            df['Range_MassWeightedMean'].fillna(0, inplace=True)

        ##MassWeightedSD
        if var2prep == 'all' or any("MassWeightedSD" in s for s in var2prep):
            if verbose:
                print 'Clean MassWeightedSD'
            df['MassWeightedSD1'] = df[['RadarCounts','MassWeightedSD']].apply(clean.getIthRadar, axis=1)
            df['Avg_MassWeightedSD'],  df['Range_MassWeightedSD'], df['Nval_MassWeightedSD']=\
              zip(*df['MassWeightedSD1'].apply(clean.getListReductions))
            df.drop('Nval_MassWeightedSD', axis=1, inplace=True)# Already in Nval
            df['Avg_MassWeightedSD'].fillna(0, inplace=True)
            df['Range_MassWeightedSD'].fillna(0, inplace=True)

        ##RhoHV
        if var2prep == 'all' or any("RhoHV" in s for s in var2prep):
            if verbose:
                print 'Clean RhoHV'
            df['RhoHV1'] = df[['RadarCounts','RhoHV']].apply(clean.getIthRadar, axis=1)
            df['Avg_RhoHV'],  df['Range_RhoHV'], df['Nval_RhoHV']=\
              zip(*df['RhoHV1'].apply(clean.getListReductions))
            df.drop('Nval_RhoHV', axis=1, inplace=True)# Already in Nval
            #df['Avg_RhoHV'].fillna(0, inplace=True)
            #df['Range_RhoHV'].fillna(0, inplace=True)


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

        self.iscleaned = True
        
    def prepare_and_save_df(self, col2save, save_name):
        """
        Prepare data and save the data frame
        """
        print '\nWill prepare and save the following column'
        print col2save
        print 'Preparing the data...'
        self.prepare_data(self.df_full, True, col2save)
        print 'Saving data...'
        self.df_full.to_hdf(save_name,'dftest',mode='w')
        print 'Done saving dataframe in {}'.format(save_name)
        
    def set_df_from_saved(self, saved_name):
        """
        sets self.df_full from saved df
        """
        self.df_full = read_hdf(saved_name, 'dftest')
        ##It is assumed to be cleaned
        self.iscleaned = True


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
            
    def __get_roc_curve(self, target_test, target_predicted_proba):
        """
        Returns a figure with roc curve
        """
        fpr, tpr, thresholds = roc_curve(target_test, target_predicted_proba[:,1])
        roc_auc = auc(fpr, tpr)  ## Area under the curve
        fig_roc = plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)'%roc_auc)
        plt.plot([0,1], [0,1], 'k--') # random prediction curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (sensitivity)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid()
        return {'fig' : fig_roc, 'auc' : roc_auc}

        
    def fitNscoreClassifier(self, col2fit, maxdepth=8, nestimators=40, **kwargs):
        """
        Fit on one fraction of the data and score on the rest
        
        Kwargs:
        showNwaite (bool): show the plots and waits for the user to press enter when finished
         default:True

        """

        if not self.iscleaned:
            print 'Preparing the data...'
            self.prepare_data(self.df_full, True, col2fit)

        test_size = 0.3## fraction kept for testing
        rnd_seed = 0## for reproducibility 

        features_train, features_test, target_train, target_test = train_test_split(
            self.df_full[col2fit].values, self.df_full['rain'].values,
            test_size=test_size, random_state=rnd_seed)

        print '\nFitting with max_depth={} and n_estimators={}...'.format(maxdepth, nestimators)
        self.rainClassifier = RandomForestClassifier(n_estimators=nestimators, max_depth=maxdepth)
        self.rainClassifier.fit(features_train, target_train)
        
        print 'Done!\n\nFeatures importances'
        ordered_feature, ordered_importance = [], []
        ord_idx = N.argsort(self.rainClassifier.feature_importances_)#Feature index ordered by importance 
        for ifeaturindex in ord_idx[::-1]:
            print '{0} \t: {1}'.format(col2fit[ifeaturindex], round(self.rainClassifier.feature_importances_[ifeaturindex], 2))
            ordered_feature.append(col2fit[ifeaturindex])
            ordered_importance.append(self.rainClassifier.feature_importances_[ifeaturindex])

        
        ## Number of cpu to use
        ## Making sure there is one free unless there is only one
        njobs = max(1, int(0.75*multiprocessing.cpu_count()))
        print '\n\nValidating with njobs = {}\n...\n'.format(njobs)


        print 'Cross validating on {} rows with njobs={}...'.format(target_test.shape[0], njobs)
        
        scores = cross_validation.cross_val_score(self.rainClassifier, features_test,
                                                  target_test, cv=10, n_jobs=njobs)
        print scores
        print '\n\nCross validation accuracy: %.2f (+/- %.3f)\n' % (round(scores.mean(), 3), round(scores.std() / 2, 3))

        ## Importances
        ordered_feature.reverse()
        ordered_importance.reverse()
        fig_importance = plt.figure(figsize = [6,9])
        y_pos = N.arange(len(ordered_feature))
        plt.barh(y_pos, ordered_importance, align='center', alpha=0.4)
        plt.yticks(y_pos, ordered_feature)
        plt.xlabel('Importance')
        #plt.tight_layout()
        plt.subplots_adjust(left=0.35, top=0.95)
        plt.grid()

        ## Probability distribution
        ## Only the max probability is shown (using N.amax(target_predicted_proba, 1))
        ## because the other probability is 1-FirstProb
        target_predicted_proba = self.rainClassifier.predict_proba(features_test)
        fig_prob = plt.figure()
        plt.hist(N.amax(target_predicted_proba, 1), normed=True, bins = 50)
        plt.xlabel('Prediction probability')
        plt.yscale('log', nonposy='clip')
        plt.grid()


        ## ROC curve
        fig_roc = plt.figure()
        roc_dic = self.__get_roc_curve(target_test, target_predicted_proba)

        waitNshow = True
        if kwargs.has_key('waitNshow'):
            waitNshow = kwargs['waitNshow']
        if waitNshow:
            fig_importance.show()
            fig_prob.show()
            roc_dic['fig'].show()
            raw_input('press enter when finished')
        out_dic =  {'fig_importance': fig_importance, 'fig_prob':fig_prob, 'fig_roc':roc_dic['fig']}
        out_dic['features_importances'] = self.rainClassifier.feature_importances_
        out_dic['scores_mean'] = scores.mean()
        out_dic['scores_std'] = scores.std()
        out_dic['roc_auc'] = roc_dic['auc']
        return out_dic

    def fitNscoreRegressor(self, col2fit, maxdepth=8, nestimators=40):
        """
        Fit the regressor only on the data with rain
        """
        if not self.iscleaned:
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
        clf_maxdepth, clf_nestimators = 18, 250
        reg_maxdepth, reg_nestimators = 13, 250
        
        combined_col = clf_col2fit + list(set(reg_col2fit) - set(clf_col2fit))
        if not self.iscleaned:
            print 'Preparing the data...'
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

        combined_col = clf_col2fit + list(set(reg_col2fit) - set(clf_col2fit))
        if not self.iscleaned:
            print 'Preparing the data...'
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
    #rfmodel = RandomForestModel(saved_df = 'saved_df/test30k.h5')
    #rfmodel = RandomForestModel(saved_df = 'saved_df/test200k.h5')
    rfmodel = RandomForestModel('Data/train_2013.csv', 700000)
    #rfmodel = RandomForestModel('Data/train_2013.csv', 'all')
    #coltofit = ['Avg_Reflectivity', 'Range_Reflectivity', 'Nval', 'Avg_RR1', 'Range_RR1', 'Avg_RR2', 'Range_RR2']
    coltofit = ['Avg_Reflectivity', 'Range_Reflectivity', 'Nval',
                'Avg_DistanceToRadar', 'Avg_RadarQualityIndex', 'Range_RadarQualityIndex',
                'Avg_RR1', 'Range_RR1','Avg_RR2', 'Range_RR2',
                'Avg_RR3', 'Range_RR3', 'Avg_Zdr', 'Range_Zdr',
                'Avg_Composite', 'Range_Composite','Avg_HybridScan', 'Range_HybridScan',
                'Avg_Velocity', 'Range_Velocity', 'Avg_LogWaterVolume', 'Range_LogWaterVolume',
                'Avg_MassWeightedMean', 'Range_MassWeightedMean',
                'Avg_MassWeightedSD', 'Range_MassWeightedSD', 'Avg_RhoHV', 'Range_RhoHV'
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
    rfmodel.prepare_and_save_df(coltofit, 'saved_df/test700k.h5')
    #rfmodel.fitNscoreAll(clf_coltofit, reg_coltofit)
    #rfmodel.submit(clf_coltofit, reg_coltofit)
    #rfmodel.fitNscoreClassifier(clf_coltofit,18, 300)
