

#!/usr/bin/python3

""" 
    this is the code to accompany the Lesson 3 (decision tree) mini-project
    use an DT to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
from sklearn import tree
from sklearn.metrics import accuracy_score

print("number of features: ", len(features_train[0]))

clf = tree.DecisionTreeClassifier(min_samples_split=40)

t0 = time()
clf = clf.fit(features_train, labels_train)
print("training time: ", time()-t0)
      
t0 = time()
pred = clf.predict(features_test)
print("prediction time: ", time()-t0)


t0 = time()
acc = accuracy_score(labels_test, pred)
print("accuracy time: ", time()-t0)
       
print("accuracy: ", acc)



#########################################################