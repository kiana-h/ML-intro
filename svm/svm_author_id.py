#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################

def author_code (numbers , predictions):
    authors_dict ={
            0: "Sara",
            1 : "Chris"
            }
    for i in numbers:
        print ("Email #" , i , " :" , authors_dict[predictions[i]])
               

def author_count (predictions):
    chris = 0
    sara = 0
    for item in predictions:
        if item==0:
            sara +=1
        elif item==1:
            chris+=1
    print ("Sara email count: " , sara)
    print ("Chris email count: " , chris)
        
    
    
def author_id(features_train, features_test, labels_train, labels_test):
    clf = SVC(kernel="rbf", C=10000)
#    features_train = features_train[:int(len(features_train)/100)] 
#    labels_train = labels_train[:int(len(labels_train)/100)] 
    
    t0 = time()
    clf.fit(features_train, labels_train)
    print("training time: ", time()-t0)
    
    t0 = time()
    pred = clf.predict(features_test)
    print("prediction time: ", time()-t0)
    
    t0 = time()
    accuracy = accuracy_score(pred, labels_test)
    print("accuracy time: ", time()-t0)
       
    print("accuracy: ", accuracy)
    
#   author_code([10,26,50] , pred)
    author_count(pred)
    
    return accuracy

author_id(features_train, features_test, labels_train, labels_test)

#########################################################

## run 1 (linear kernel, complete data set)
#training time:  196.89907813072205
#prediction time:  20.625083923339844
#accuracy time:  0.0009984970092773438
#accuracy: 0.9840728100113766

## run 2 (linear kernel, 1% of data set)
#training time:  0.08693671226501465
#prediction time:  0.8864727020263672
#accuracy time:  0.0010001659393310547
#accuracy:  0.8845278725824801

##run 3 (rbf kernel, 1% of data set)
#training time:  0.12592864036560059
#prediction time:  1.3480229377746582
#accuracy time:  0.0019829273223876953
#accuracy:  0.6160409556313993

##run 4 (rbf kernel, 1% of data set, C=10)
#training time:  0.12406778335571289
#prediction time:  1.3177440166473389
#accuracy time:  0.0019822120666503906
#accuracy:  0.6160409556313993

##run 4 (rbf kernel, 1% of data set, C=100)
#training time:  0.11994552612304688
#prediction time:  1.252777099609375
#accuracy time:  0.0010004043579101562
#accuracy:  0.8213879408418657

##run 5 (rbf kernel, 1% of data set, C=1000)
#training time:  0.12094330787658691
#prediction time:  1.2552785873413086
#accuracy time:  0.0009868144989013672
#accuracy:  0.8213879408418657

##run 6 (rbf kernel, 1% of data set, C=10000)
#training time:  0.11594891548156738
#prediction time:  1.061931848526001
#accuracy time:  0.0
#accuracy:  0.8924914675767918

##run 7 (rbf kernel, full data set, C=10000)
#training time:  123.00739765167236
#prediction time:  12.883005380630493
#accuracy time:  0.0009865760803222656
#accuracy:  0.9908987485779295
