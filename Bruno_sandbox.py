#================================================================================
#
# This is my sandbox; I play here. 
#
#================================================================================

## import project modules

from rf2steps import *
import clean


# Read in the data and clean it

nrows = 100 # just to play
data_filename = './Data/train_2013.csv'

rfmodel = RandomForestModel(train_data_fname=data_filename, nrows=nrows)

input_df = rfmodel.df_full 

output_df = clean.get_dataframe_with_split_multiple_radars(input_df)
#rfmodel.prepare_data(df, verbose = True, var2prep = 'all')

