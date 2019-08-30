#!/usr/bin/python3

import sys
import pickle
from time import time
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn import cross_validation
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

import matplotlib.pyplot as plt

dash = 40
file_loc = "final_project_dataset_unix.pkl"
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#features_list = ['poi','salary','bonus', 'long_term_incentive','to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']


def load_data(file):
    ### Load the dictionary containing the dataset
    print("loading data...")
    data_dict = pickle.load(open(file , "rb"))
    print("data loaded!")
    print("-"*dash)
    return data_dict

dataset = load_data(file_loc)
    
def explore_data(dataset):
    """
    prints some basic info about the dataset
    input: dataset location 
    """      
    data_points = len(dataset)
    print("total number of data points: ", data_points)
    print("-"*dash)
    
    ### Get Number of POI/Non-POI
    poi_no = 0
    for person in dataset.values():
        if person["poi"]:
            poi_no +=1
    print("allocation across classes (POI/non-POI): ")
    print("number of POI:", poi_no)
    print("number of non-POI: ", data_points-poi_no)
    print("-"*dash)
    
    ### Get info on features
    feature_sample = list((list(dataset.values())[0]).keys())
    print("number of features in dataset: ",len(feature_sample))
    print("dataset_features: ", feature_sample)
    print("-"*dash)
    
    return dataset, len(feature_sample)

 
def remove_incomplete(data, percentage):
    """
    removes datapoints that have more than specified percentage missing features
    input: dataset, percentage(0.8)
    output: dataset without incomplete datapoints
    """
    dataset, feature_count = explore_data(data)  
    #create list of people with complete data
    new_list = {}
    for person, feature_list in dataset.items():
        missing_features = 0
        for feature in feature_list.values():
            if feature == 'NaN':
                missing_features += 1
        if missing_features/feature_count < percentage:
            new_list[person] = feature_list
    print("removed %d people" %(len(dataset) - len(new_list)))
    explore_data(new_list)
    
    return new_list
        
            
my_dataset = remove_incomplete(dataset, .7)

#### Task 2: Remove outliers

def data_plot(dataset,features):
    """visualize data based on key features"""
    data = featureFormat(my_dataset, features)
    for point in data:
        salary = point[0]
        bonus = point[1]
        plt.scatter( salary, bonus )
    plt.xlabel("salary")
    plt.ylabel("bonus")
    plt.show()

### we plot the data based on key features (salary and bonus) to detect outliers    
key_features = ["salary", "bonus"]
print("dataset before removing outliers: ")
data_plot(my_dataset , key_features) 
### remove outlier (refer to enron_outlier file)   
my_dataset.pop("TOTAL")
print("dataset after removing outliers: ")
data_plot(my_dataset , key_features)  
print("-"*dash)


#### Task 3: Create new feature(s)
#### feature categories
#financial_f = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
#email_f =  ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
#poi_f = ['poi']
 
def email_analysis(dataset):
    for person,features in dataset.items():
        try:
            dataset[person]["from_poi_percent"] = features["to_messages"]/features["from_poi_to_this_person"]
        except:
            dataset[person]["from_poi_percent"] = "NaN"
        try:
            dataset[person]["to_poi_percent"] = features["from_this_person_to_poi"]/features["to_messages"]
        except:
            dataset[person]["to_poi_percent"] = "NaN"
    
    return(dataset)

email_analysis(my_dataset)       
features_list = ['poi','salary','deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options','long_term_incentive', 'restricted_stock', 'director_fees', 'from_poi_percent', 'to_poi_percent']    

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
print(data)
labels, features = targetFeatureSplit(data)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(features)

features= SelectKBest(f_classif, k=4).fit_transform(features, labels)
print(features)
#features_train, features_test, labels_train, labels_test = cross_validation.train_test_split (features, labels, test_size=0.4)


#pca = PCA(n_components=3)
#pca.fit(features_train)
#X_train_pca = pca.transform(features_train)
#X_test_pca = pca.transform(features_test)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

def test_clf(labels, features, parameters, typ, name):   
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split (features, labels, test_size=0.3)
    time0 = time()
    clf = GridSearchCV(typ, parameters)
    clf.fit(features_train,labels_train)
    clf_best= clf.best_estimator_
    pred = clf_best.predict(features_test)
    accuracy = round(accuracy_score(pred, labels_test),3)
    time1 = time()
    calc_time = round(time1-time0 , 2)
    print("best accuracy using", name, ": ", accuracy)
    print("calc time using", name, ": ", calc_time)
    print("optimized paramaters for", name, ": ", clf.best_params_)
    print("-"*dash)
    return(accuracy, calc_time)
    
def test_svm(labels, features):
    parameters = {'kernel':['rbf'], 'C':[1, 10, 100, 1000], }
    svr = svm.SVC()
    return test_clf(labels, features, parameters, svr, "svm")
    
def test_dt(labels, features): 
    parameters = {'criterion':['gini','entropy'], 'min_samples_split':[4, 8, 16, 32, 64], 'max_features': ['auto', 'sqrt', 'log2', None]}
    dt = tree.DecisionTreeClassifier()
    return test_clf(labels, features, parameters, dt, "decision tree")
    
def test_knn(labels, features): 
    parameters = {'n_neighbors':[2, 4, 8, 16]}
    knn = KNeighborsClassifier()
    return test_clf(labels, features, parameters, knn, "knn")    
    
def test_rf(labels, features): 
    parameters = {'n_estimators': [5,10,20] , 'criterion':['gini','entropy'], 'min_samples_split':[4, 8, 16, 32, 64]}
    rf = RandomForestClassifier()
    return test_clf(labels, features, parameters, rf, "random forest") 
    
def test_ab(labels, features): 
    parameters = {'n_estimators': [5,10,20,50,80] , 'learning_rate': [0.25 , 0.5 , 1 , 2]}
    ab = AdaBoostClassifier()
    return test_clf(labels, features, parameters, ab, "adaboost") 

        
def test_clfs(labels, features):
    test_svm(labels, features)
    test_dt(labels, features)
    test_knn(labels, features)
    test_rf(labels, features)
    test_ab(labels, features)
    
def pick_clf(labels, features):
    svm = []
    dt = []
    knn = []
    for i in range(100):
        svm.append(test_svm(labels, features))
        dt.append(test_dt(labels, features))
        knn.append(test_knn(labels, features))
        
    svm = sum([x[0] for x in svm])/len(svm)
    dt = sum([x[0] for x in dt])/len(dt)
    knn = sum([x[0] for x in knn])/len(knn)
    
    print("average accuracy using svm: ", round(svm,2))
    print("average accuracy using dt: ", round(dt,2))
    print("average accuracy using knn: ", round(knn,2))
        
pick_clf(labels, features)    

#### Task 5: Tune your classifier to achieve better than .3 precision and recall 
#### using our testing script. Check the tester.py script in the final project
#### folder for details on the evaluation method, especially the test_classifier
#### function. Because of the small size of the dataset, the script uses
#### stratified shuffle split cross validation. For more info: 
#### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
#
## Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)
#
#### Task 6: Dump your classifier, dataset, and features_list so anyone can
#### check your results. You do not need to change anything below, but make sure
#### that the version of poi_id.py that you submit can be run on its own and
#### generates the necessary .pkl files for validating your results.
#
dump_classifier_and_data(clf, my_dataset, features_list)