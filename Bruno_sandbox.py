#================================================================================
#
# This is my sandbox; I play here. 
#
#================================================================================

## import project modules

from rf2steps import *
import clean


# Read in the data and clean it

create_data = False
#hdf5_filename = 'sandbox.h5' 
hdf5_filename = 'sandbox_fast.h5' 

if create_data:
    nrows = 10000 # just to play
    #nrows = 'all'  
    data_filename = './Data/train_2013.csv'

    rfmodel   = RandomForestModel(train_data_fname=data_filename, nrows=nrows)
    input_df  = rfmodel.df_full 
    output_df = clean.get_dataframe_with_split_multiple_radars(input_df)

    # write to file
    output_df.to_hdf(hdf5_filename ,'dftest',mode='w')
else:
    output_df =  pd.read_hdf(hdf5_filename,'dftest')

expected = output_df['Expected'].values


ranges = [ [ -0.1,0.1],[0.1,2.5], [2.5,10], [10,70] ]

list_labels = []
list_values = []

#key = 'Velocity'
#key = 'HydrometeorType'
#key = 'Zdr'
#key = 'LogWaterVolume'
#key = 'RhoHV'
#key = 'RR1'
#key = 'RadarQualityIndex'
key = 'Reflectivity'
for range in ranges:
    condition = output_df['Expected'].apply(lambda n: n > range[0] and n <= range[1] )

    values    = output_df[condition][key].apply(N.average).values
    list_values.append(values)

    list_labels.append(' %2.1f < rain <= %2.1f'%(range[0],range[1]))


hist = plt.hist(list_values,bins=100,normed=True, stacked=False, histtype='stepfilled',
                label=list_labels,alpha=0.75)
plt.xlabel('Average %s'%key)
plt.ylabel('Probability density')
plt.title('Distribution of average %s, depending on whether it rained or not'%key)
plt.legend(loc=0)

plt.show()

#================================================================================
# I want to test and plot the classifier with a varying parameter 
# which determines the split between "rain" and "no rain". I want to produce
# nice plots like ones coming out of the use of "validation_curve". I just can't
# see straight right now, so it'll have to wait until later.
#================================================================================

nestimators = 10
max_depth   = 8

forest = RandomForestClassifier(n_estimators=nestimators, max_depth=maxdepth)



"""
# hack that data back into the model objecti
rfmodel.df_full = output_df
var2prep  = ['Reflectivity']

rfmodel.prepare_data(rfmodel.df_full, verbose = True, var2prep = 'all')

#rfmodel.prepare_data(df, verbose = True, var2prep = 'all')
"""
