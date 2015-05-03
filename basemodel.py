"""Base class for models to be trained."""
import pandas as pd
# modules from this projects
import clean


class BaseModel(object):
    """Contain members and function for all trainers."""
    def __init__(self, train_data_fname=None, nrows='all', **kwargs):
        """Turn data in pandas dataframe."""
        verbose = kwargs.get('verbose', True)
        # Define the classifier and regressor variables
        self.iscleaned = False
        self.df_full = None

        if 'saved_df' in kwargs.keys():
            print('Get saved train data from {}'.format(kwargs['saved_df']))
            self.df_full = pd.read_hdf(kwargs['saved_df'], 'dftest')
            print('Train data frame has shape')
            print(self.df_full.shape)
            self.iscleaned = True
            return

        if train_data_fname is None:
            print 'Data will not be created by reading csv file'
            self.df_full is None
            return

        if nrows == 'all':
            self.df_full = pd.read_csv(train_data_fname)
        else:
            self.df_full = pd.read_csv(train_data_fname, nrows=nrows)

        if verbose:
            print('Training data frame has shape')
            print(self.df_full.shape)

    def clean_data(self, df, verbose=False, var2prep='all'):
        """Prepare self.df_full for fitting.

        var2prep is a list of variables that will be needed.
        This will save time by cleaning only the needed variables
        """
        if self.iscleaned:
                print('Data already cleaned')
                return

        if verbose:
            print('Getting radar length')
        df['RadarCounts'] = df['TimeToEnd'].apply(clean.getRadarLength)

        # Drop rows where the expected rain is above 70
        # This will also exclude them for the scoring (our scoring, not the kaggle one)
        if 'Expected' in df.columns.values:
            df.drop(df[df['Expected']>70].index, inplace=True)

        # Add a category column rain/norain (1/0)
        # Might consider using a threshold i.e. rain if Expected > threshold
        if 'Expected' in df.columns.values:
            df['rain'] = df['Expected'].apply(lambda x: 1 if x>0 else 0)

        # #####
        # Adding the mean of variables to fit

        # Reflectivity
        if var2prep == 'all' or any("Reflectivity" in s for s in var2prep):
            if verbose:
                print 'Clean reflectivity'
            df['Reflectivity1'] = df[['RadarCounts','Reflectivity']].apply(
                clean.getIthRadar, axis=1)
            df['Avg_Reflectivity'],  df['Range_Reflectivity'], df['Nval']=\
              zip(*df['Reflectivity1'].apply(clean.getListReductions))

        # Zdr
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

        # Composite
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

        # HybridScan
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

        # Velocity
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

        # LogWaterVolume
        if var2prep == 'all' or any("LogWaterVolume" in s for s in var2prep):
            if verbose:
                print 'Clean LogWaterVolume'
            df['LogWaterVolume1'] = df[['RadarCounts','LogWaterVolume']].apply(clean.getIthRadar, axis=1)
            df['Avg_LogWaterVolume'],  df['Range_LogWaterVolume'], df['Nval_LogWaterVolume']=\
              zip(*df['LogWaterVolume1'].apply(clean.getListReductions))
            df.drop('Nval_LogWaterVolume', axis=1, inplace=True)# Already in Nval
            df['Avg_LogWaterVolume'].fillna(0, inplace=True)
            df['Range_LogWaterVolume'].fillna(0, inplace=True)

        # MassWeightedMean
        if var2prep == 'all' or any("MassWeightedMean" in s for s in var2prep):
            if verbose:
                print 'Clean MassWeightedMean'
            df['MassWeightedMean1'] = df[['RadarCounts','MassWeightedMean']].apply(clean.getIthRadar, axis=1)
            df['Avg_MassWeightedMean'],  df['Range_MassWeightedMean'], df['Nval_MassWeightedMean']=\
              zip(*df['MassWeightedMean1'].apply(clean.getListReductions))
            df.drop('Nval_MassWeightedMean', axis=1, inplace=True)# Already in Nval
            df['Avg_MassWeightedMean'].fillna(0, inplace=True)
            df['Range_MassWeightedMean'].fillna(0, inplace=True)

        # MassWeightedSD
        if var2prep == 'all' or any("MassWeightedSD" in s for s in var2prep):
            if verbose:
                print 'Clean MassWeightedSD'
            df['MassWeightedSD1'] = df[['RadarCounts','MassWeightedSD']].apply(clean.getIthRadar, axis=1)
            df['Avg_MassWeightedSD'],  df['Range_MassWeightedSD'], df['Nval_MassWeightedSD']=\
              zip(*df['MassWeightedSD1'].apply(clean.getListReductions))
            df.drop('Nval_MassWeightedSD', axis=1, inplace=True)# Already in Nval
            df['Avg_MassWeightedSD'].fillna(0, inplace=True)
            df['Range_MassWeightedSD'].fillna(0, inplace=True)

        # RhoHV
        if var2prep == 'all' or any("RhoHV" in s for s in var2prep):
            if verbose:
                print 'Clean RhoHV'
            df['RhoHV1'] = df[['RadarCounts','RhoHV']].apply(clean.getIthRadar, axis=1)
            df['Avg_RhoHV'],  df['Range_RhoHV'], df['Nval_RhoHV']=\
              zip(*df['RhoHV1'].apply(clean.getListReductions))
            df.drop('Nval_RhoHV', axis=1, inplace=True)# Already in Nval
            #df['Avg_RhoHV'].fillna(0, inplace=True)
            #df['Range_RhoHV'].fillna(0, inplace=True)


        # Distance to radar
        if var2prep == 'all' or any("DistanceToRadar" in s for s in var2prep):
            if verbose:
                print 'Clean DistanceToRadar'
            df['DistanceToRadar1'] = df[['RadarCounts','DistanceToRadar']].apply(clean.getIthRadar, axis=1)
            df['Avg_DistanceToRadar'],  df['Range_DistanceToRadar'], df['Nval_DistanceToRadar']=\
              zip(*df['DistanceToRadar1'].apply(clean.getListReductions))
            ## Remove the Nval_xxx it's already in the Nval column
            df.drop('Nval_DistanceToRadar', axis=1, inplace=True)

        # Radar quality index
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

        # RR1
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

        # RR2
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

        # RR3
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

        # HydrometeorType
        # Hydrometers are categories so we create variables for each one
        # The column contains how many times the type occurs
        # This could be scaled by nval to keep the value with [0,1]
        # but since nval is also a feature, I let the learner deal with it
        if var2prep == 'all' or any("hm_" in s for s in var2prep):
            if verbose:
                print 'Clean HydrometeorType'
            df['HyMeType1'] = df[['RadarCounts','HydrometeorType']].apply(clean.getIthRadar, axis=1)
            for itype in range(15):
                df['hm_{}'.format(itype)] = df.HyMeType1.apply(lambda x: x.count(itype))
            # Some values have the same meaning let's add them
            df['hm_0'] = df['hm_0'] + df['hm_9']
            df.drop('hm_9', axis=1, inplace=True)

            df['hm_1'] = df['hm_1'] + df['hm_2']
            df.drop('hm_2', axis=1, inplace=True)

            df['hm_13'] = df['hm_13'] + df['hm_14']
            df.drop('hm_14', axis=1, inplace=True)

            df.drop('HyMeType1', axis=1, inplace=True)

        self.iscleaned = True
