#================================================================================
#
# Class for random forest training
#
#================================================================================

import pandas as pd
import clean
import matplotlib.pyplot as plt

feature_ranges = {'Avg_Reflectivity': [-10, 50]}


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
        df['Reflectivity1'] = df[['RadarCounts','Reflectivity']].apply(
            clean.getIthRadar, axis=1)
        df['Avg_Reflectivity'],  df['Range_Reflectivity'], df['Nval_Reflectivity']=\
          zip(*df['Reflectivity1'].apply(clean.getListReductions))


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

        ## Plot
        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.hist([rain_feature,norain_feature],
               bins=100,normed=True,stacked=False,histtype='stepfilled',
                label=['Rain','No Rain'],alpha=0.75)
        plt.legend(loc='best')
        plt.title(feature)
        plt.subplot(1,2,2)
        #plt.scatter(self.df_train[rain][feature].get_values(), self.df_train[rain]['Expected'].get_values(), color='Red', label='Rain')
        ranges = [[min(rain_feature), max(rain_feature)],[0, 20]]
        if feature_ranges.has_key(feature):
                ranges[0] = feature_ranges[feature]
        plt.hist2d(self.df_train[rain][feature].get_values(), self.df_train[rain]['Expected'].get_values(), range=ranges, bins=100, norm=LogNorm())
        plt.xlabel(feature)
        plt.ylabel('Rain gauge')
        fig.show()
        raw_input('press enter when finished...')



                
if __name__=='__main__':
    rfmodel = RandomForestModel('Data/train_2013.csv', 20000)
    rfmodel.show_feature('Avg_Reflectivity')
