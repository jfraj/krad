#================================================================================
#
# This is my sandbox; I play here. 
#
#================================================================================

# import project modules
import time

from functools import partial

from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier                                                                              


from sklearn.svm import LinearSVC, SVC

from sklearn.decomposition import PCA
from rf2steps import *
import clean
import seaborn; seaborn.set(font_scale=2)


# Read in the data and clean it
#create_data = True
create_data = False
#hdf5_filename = 'sandbox.h5' 
hdf5_filename = 'sandbox_fast.h5' 

if create_data:
    nrows = 10000  # just to play
    #nrows = 'all'  
    data_filename = './Data/train_2013.csv'

    rfmodel   = RandomForestModel(train_data_fname=data_filename, nrows=nrows)
    input_df  = rfmodel.df_full 
    split_df  = clean.get_dataframe_with_split_multiple_radars(input_df)

    #  drop senseless data points
    clean_split_df = split_df[split_df['Expected'] <=70.]

    #output_df = clean.get_clean_average_dataframe(clean_split_df)

    output_df = clean.get_clean_average_and_range_dataframe(clean_split_df)

    # write to file
    output_df.to_hdf(hdf5_filename ,'dftest',mode='w')
else:
    output_df =  pd.read_hdf(hdf5_filename,'dftest')

#================================================================================


def get_class_threshold(expected,threshold):
    if expected >= threshold:
        return 1.
    else:
        return 0.


nmax = 50000
avg_df = output_df[:nmax].drop(['Id','Expected','number_of_radars'],1)

X  = avg_df.values


list_theta = N.arange(0.01,5.,1.)
list_no_rain = []
list_correct = []

LR = LogisticRegression()
P  = Perceptron()
NSVC = SVC()
LSVC = LinearSVC()

n_estimators = 600
max_depth    = 15
RFC  = RandomForestClassifier( n_estimators=n_estimators,  max_depth=max_depth )


name_dict = {'Logistic Regression':'LR',
                      'Perceptron': 'Perceptron',
                   'Support Vector': 'SVC',
            'Linear Support Vector': 'LSVC',
         'Random Forest classifier': 'RFC'}

Model_dict = {'Logistic Regression':[LR,[],[]],
                       'Perceptron':[P,[],[]],
                   'Support Vector':[NSVC,[],[]],
            'Linear Support Vector':[LSVC,[],[]],
         'Random Forest classifier':[RFC,[],[]]}


#Model_dict = {'Logistic Regression':[LR,[],[]]}
#Model_dict = { 'Support Vector':[NSVC,[],[]]}




for theta in list_theta:

    get_class = partial(get_class_threshold, threshold = theta)

    t = output_df[:nmax]['Expected'].apply(get_class).values

    no_rain_ratio  = 1.*len(N.where(t == 0)[0])/len(t)
    list_no_rain.append(no_rain_ratio)
    print 'fraction of no rain          : %6.4f '%no_rain_ratio

    for key in Model_dict.keys():

        M = Model_dict[key][0]

    
        time1 = time.time()            
        M.fit(X,t)
        time2 = time.time()            
        delta_time = time2-time1

        t_fit = M.predict(X)

        I_right = N.where(t == t_fit)[0]
        I_wrong = N.where(t != t_fit)[0]

        right = 1.*len(I_right)
        wrong = 1.*len(I_wrong)

        right_ratio = right/(right+wrong)
        Model_dict[key][1].append(right_ratio) 
        Model_dict[key][2].append(delta_time) 

list_no_rain = N.array(list_no_rain )


w = 16
h = 8
fig = plt.figure(figsize=(w,h))
ax1  = fig.add_subplot(121)
ax2  = fig.add_subplot(122)


ax1.plot(list_theta,100*list_no_rain,lw=4,label='Fraction of low rain data')
ax1.fill_between(list_theta,100*list_no_rain,0,alpha=0.5)

i = 0
list_bar_x      = []
list_bar_labels = []
for key in Model_dict.keys():

    list_correct = N.array(Model_dict[key][1])
    list_times   = N.array(Model_dict[key][2])

    lines, = ax1.plot(list_theta,100*list_correct,'-o',ms=10,label='%s'%key)

    color = lines.get_color()

    #ax2.semilogy(list_theta,list_times,'-o',c=color,ms=10,label='%s'%key)

    ax2.bar(i,N.mean(list_times),color=color)

    list_bar_labels.append(name_dict[key])
    list_bar_x.append(i+0.4)
    i += 1


ax1.set_title('Testing Classifiers with fitting data')
ax1.set_xlabel('Threshold value to rain (mm)')
ax1.set_ylabel('% of correct prediction')

#ax2.set_xlabel('Threshold value to rain (mm)')
ax2.set_xticks(list_bar_x)
ax2.set_xticklabels(list_bar_labels)

ax2.set_ylabel('Approx. execution time (sec)')
ax2.set_yscale('log')


ax1.set_xlim([0,4])

ax1.set_ylim([80,100])
ax1.legend(loc=0)
plt.savefig('comparing_classifiers.pdf')
#plt.show()

