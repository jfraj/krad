#================================================================================
#
# This is my sandbox; I play here. 
#
#================================================================================

## import project modules
import time

import multiprocessing

from functools import partial

from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier                                                                              

from sklearn.learning_curve import learning_curve                                                                       


from sklearn.svm import LinearSVC, SVC

from sklearn.decomposition import PCA
from rf2steps import *
import clean
import seaborn; seaborn.set(font_scale=2)

# Read in the data and clean it
hdf5_filename = 'sandbox_fast.h5' 
output_df =  pd.read_hdf(hdf5_filename,'dftest')

#================================================================================

def get_class_threshold(expected,threshold):
    if expected > threshold:
        return 1.
    else:
        return 0.

get_class = partial(get_class_threshold, threshold = 0.0)


njobs = max(1, multiprocessing.cpu_count()-1)
njobs = 1

print 'njob = %i'%njobs
nmax = 50000

t  = output_df[:nmax]['Expected'].apply(get_class).values

avg_df = output_df[:nmax].drop(['Id','Expected','number_of_radars'],1)
X      = avg_df.values


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
         'Random Forest classifier (n=600, depth=15)':[RFC,[],[]]}

Model_dict = {'Logistic Regression':[LR,[],[]],
                       'Perceptron':[P,[],[]],
            'Linear Support Vector':[LSVC,[],[]],
         'Random Forest classifier (n=600, depth=15)':[RFC,[],[]]}


#Model_dict = { 'Random Forest classifier (n=600, depth=15)':[RFC,[],[]]}



w = 16
h = 8
ms = 10
fig = plt.figure(figsize=(w,h))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

fig.suptitle('Learning Curves for various algorithms')

ax1.set_title('Train scores')
ax2.set_title('Test scores')

for ax in [ax1,ax2]:
    ax.set_xlabel('training set size')
    ax.set_ylabel('score')
    ax.set_ylim([0.3,1.])


for key in Model_dict.keys(): 

    print 'doing %s...'%key
    estimator = Model_dict[key][0]

    train_sizes_abs, train_scores, test_scores = \
        learning_curve(estimator, X, t, train_sizes = N.array([ 0.1  ,  0.325,  0.55 ,  0.775,  1.   ]),n_jobs=njobs,verbose=2)

    train_scores_mean = N.mean(train_scores, axis=1)
    train_scores_std  = N.std(train_scores, axis=1)

    test_scores_mean = N.mean(test_scores, axis=1)
    test_scores_std = N.std(test_scores, axis=1)

    lines, = ax1.plot(train_sizes_abs, train_scores_mean, 'o-', ms=ms, label=key)

    color  = lines.get_color()
    ax1.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color=color)

    lines, = ax2.plot(train_sizes_abs, test_scores_mean, 'o-', ms=ms, label=key)
    color  = lines.get_color()

    ax2.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color=color)


for ax in [ax1,ax2]:
    labels = ax.get_xticklabels() 
    for label in labels: 
            label.set_rotation(90) 


ax1.legend(loc="best")
plt.tight_layout()
#ax1.set_xlim([0,4])
#ax1.set_ylim([80,100])
#ax1.legend(loc=0)
plt.savefig('comparing_learning.pdf')
#plt.show()
