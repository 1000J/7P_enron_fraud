#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tester
from time import time
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data



# scatter plots of each feature versus poi
def plot_scatter(dict, feat_list):

    for feature in feat_list:
        for key in dict:
            x = dict[key][feature]
            y = dict[key]['poi']

#            if (x != 'NaN'):
#                plt.scatter( x, y )
            
            if (x == 'NaN'):
                plt.scatter(0,y)
            else:
                plt.scatter( x, y )
        plt.xlabel(feature)
        plt.ylabel('poi')
        plt.show()

    return None



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','deferral_payments', 
                 'total_payments', 'loan_advances', 'bonus', 
                 'restricted_stock_deferred', 'deferred_income', 
                 'total_stock_value', 'expenses', 'exercised_stock_options', 
                 'other', 'long_term_incentive', 'restricted_stock', 
                 'director_fees',
                 'to_messages', 'from_poi_to_this_person', 'from_messages', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)

#for key in data_dict:
#    print key, data_dict[key]['poi']
    
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# scatter plot of features versus poi to visualize data
# also provides additional check for outliers
# highlighted out because of space it takes
plot_scatter(my_dataset, features_list)

# add two new engineered features 
for key in my_dataset:
#    print key
    if (my_dataset[key]['long_term_incentive']=='NaN' or \
        my_dataset[key]['total_payments']=='NaN'):
        my_dataset[key]['ratio_lt_incentive_to_tot_payments'] = 0
    else:
        my_dataset[key]['ratio_lt_incentive_to_tot_payments'] = \
            float(my_dataset[key]['long_term_incentive'])/ \
            float(my_dataset[key]['total_payments'])

    if (my_dataset[key]['restricted_stock']=='NaN' or \
        my_dataset[key]['total_stock_value']=='NaN'):
        my_dataset[key]['ratio_rstock_to_tstock'] = 0
    else:
        my_dataset[key]['ratio_rstock_to_tstock'] = \
            float(my_dataset[key]['restricted_stock'])/ \
            float(my_dataset[key]['total_stock_value'])


# update features_list to include the new engineered features
features_list = ['poi','salary','deferral_payments', 
                 'total_payments', 'loan_advances', 'bonus', 
                 'restricted_stock_deferred', 'deferred_income', 
                 'total_stock_value', 'expenses', 'exercised_stock_options', 
                 'other', 'long_term_incentive', 'restricted_stock', 
                 'director_fees',
                 'to_messages', 'from_poi_to_this_person', 'from_messages', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi',
                 'ratio_lt_incentive_to_tot_payments',
                 'ratio_rstock_to_tstock']


### Extract features and labels from dataset for local testing
# make remove_all_zeroes=False
#data = featureFormat(my_dataset, features_list, sort_keys = True, 
#                     remove_NaN = True)
data = featureFormat(my_dataset, features_list, sort_keys = True,
                     remove_all_zeroes = False) 

labels, features = targetFeatureSplit(data)

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1,test_size=44,random_state=42)

for train, test in sss.split(features,labels):
    print(len(train),len(test))
    #print train
    #print test

#features_train = 1



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedShuffleSplit



# function that runs GridSearchCV
def runGSCV(pipeline, parameters, cv, flist):
    
    gs = GridSearchCV(pipeline, param_grid=parameters, cv=cv, scoring='f1')
    
    t0=time()
    gs.fit(features, labels)
    print 'done in %0.3fs' % (time() - t0)

    print 'best score %3.3f' % gs.best_score_
    print 'best params: %s' % gs.best_params_
    gs_scores = gs.grid_scores_

    #for key in gs_scores:
    #    print key

    print '------- best features'

    best_features = sorted(zip(flist[1:], 
                               gs.best_estimator_.steps[0][1].scores_,
                               gs.best_estimator_.steps[0][1].get_support()),
                        reverse=True, key=lambda x:x[1])
    #print best_features    

    for list in best_features:
        if list[2]==True:
            print list

    print '----- running tester'
    dump_classifier_and_data(gs.best_estimator_.steps[1][1], my_dataset, flist)
    tester.main()

    return None


    
##############################################################
print '*** GridSearchCV with StratifiedShuffleSplit using DTC & SelectKBest'

#best_cv=StratifiedShuffleSplit(labels,n_iter=100,random_state=42, test_size=0.2)
best_cv=StratifiedShuffleSplit(labels,n_iter=100,random_state=42, test_size=0.3)
pipeline = Pipeline([('kbest', SelectKBest()),('dtc', DecisionTreeClassifier())])
parameters = {"kbest__k": [1, 2, 3, 4, 5, 6, 8, 10, 15, 20],
              'dtc__criterion': ['gini', 'entropy'], 
              'dtc__max_depth': [None, 1, 2, 5, 10, 20, 50],
              'dtc__max_features': [None, 'auto', 'log2']}
# fake parameter to run fast
#parameters = {"kbest__k": [1, 2],
#              'dtc__criterion': ['gini'], 
#              'dtc__max_depth': [None],
#              'dtc__max_features': ['auto']}

runGSCV(pipeline, parameters, best_cv, features_list)


##############################################################
print '*** GridSearchCV with StratifiedShuffleSplit using GNB & SelectKBest'

#best_cv=StratifiedShuffleSplit(labels,n_iter=100,random_state=42, test_size=0.2)
best_cv=StratifiedShuffleSplit(labels,n_iter=100,random_state=42, test_size=0.3)
pipeline = Pipeline([('kbest', SelectKBest()),('gnb', GaussianNB())])
parameters = {"kbest__k": [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]}

runGSCV(pipeline, parameters, best_cv, features_list)

##############################################################
print '*** GridSearchCV with StratifiedShuffleSplit using AdaBoost & SelectKBest'

#best_cv=StratifiedShuffleSplit(labels,n_iter=100,random_state=42, test_size=0.2)
best_cv=StratifiedShuffleSplit(labels,n_iter=100,random_state=42, test_size=0.3)
pipeline = Pipeline([('kbest', SelectKBest()),('abc', AdaBoostClassifier())])
parameters = {"kbest__k": [1, 2, 3, 4, 5, 6, 8, 10, 15, 20],
              'abc__n_estimators': [5,25,50,100,200], 
              'abc__algorithm': ['SAMME', 'SAMME.R'],
              'abc__random_state': [42]}
# fake parameter to run fast
#parameters = {"kbest__k": [10],
#              'abc__n_estimators': [100,], 
#              'abc__algorithm': ['SAMME'],
#              'abc__random_state': [42]}

runGSCV(pipeline, parameters, best_cv, features_list)










### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.






dump_classifier_and_data(clf, my_dataset, features_list)