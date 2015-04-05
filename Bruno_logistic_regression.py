#================================================================================
#
# This is my sandbox; I play here. 
#
#================================================================================

## import project modules

from functools import partial

from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import LinearSVC, SVC

from sklearn.decomposition import PCA
from rf2steps import *
import clean
import seaborn; seaborn.set(font_scale=2)


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
    split_df  = clean.get_dataframe_with_split_multiple_radars(input_df)

    #  drop senseless data points
    clean_split_df = split_df[split_df['Expected'] <=70.]


    output_df = clean.get_clean_average_dataframe(clean_split_df)

    # write to file
    output_df.to_hdf(hdf5_filename ,'dftest',mode='w')
else:
    df =  pd.read_hdf(hdf5_filename,'dftest')

#================================================================================

partial

def get_class_threshold(expected,threshold):
    if expected >= threshold:
        return 1.
    else:
        return 0.


avg_df = df.drop(['Id','Expected','number_of_radars'],1)

X  = avg_df.values

"""
pca = PCA(n_components=2)
pca.fit(Data)
X = pca.transform(Data)
"""

list_theta = N.arange(0.01,5.,1.)
list_no_rain = []
list_correct = []

LR = LogisticRegression()
P  = Perceptron()
NSVC = SVC()
LSVC = LinearSVC()

Model_dict = {'Logistic Regression':[LR,[]],
                       'Perceptron':[P,[]],
                   'Support Vector':[NSVC,[]],
            'Linear Support Vector':[LSVC,[]]}

#Model_dict = {'Logistic Regression':[LR,[]]}




for theta in list_theta:

    get_class = partial(get_class_threshold, threshold = theta)

    t = df['Expected'].apply(get_class).values

    no_rain_ratio  = 1.*len(N.where(t == 0)[0])/len(t)
    list_no_rain.append(no_rain_ratio)
    print 'fraction of no rain          : %6.4f '%no_rain_ratio

    for key in Model_dict.keys():

        M = Model_dict[key][0]

        M.fit(X,t)

        t_fit = M.predict(X)

        I_right = N.where(t == t_fit)[0]
        I_wrong = N.where(t != t_fit)[0]

        right = 1.*len(I_right)
        wrong = 1.*len(I_wrong)

        right_ratio = right/(right+wrong)
        Model_dict[key][1].append(right_ratio) 

list_no_rain = N.array(list_no_rain )


w = 12
h = 8
fig = plt.figure(figsize=(w,h))
ax1  = fig.add_subplot(111)

#ax1.plot(list_theta,100*list_no_rain,'-',label='Fraction of low rain data')
ax1.fill_between(list_theta,100*list_no_rain,0,alpha=0.5,label='Fraction of low rain data')

for key in Model_dict.keys():

    list_correct = N.array(Model_dict[key][1])
    ax1.plot(list_theta,100*list_correct,'-o',label='%s'%key)

ax1.set_xlabel('Threshold value to rain (mm)')
ax1.set_ylabel('% of dataset')
ax1.set_title('Various classifier models')

for ax in [ax1]:
    ax.legend(loc=0)
    ax.set_ylim([80,100])
    ax.set_xlim([0,4])

plt.show()
