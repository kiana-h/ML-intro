#!/usr/bin/python3

import pickle
import numpy
numpy.random.seed(42)


### the words (features) and authors (labels), already largely processed
### these files should have been created from the previous (Lesson 10) mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = ""
authors = ""

with open(words_file, "rb") as f:
    word_data = pickle.load(f)
with open(authors_file, "rb") as f:
    authors = pickle.load(f)


### test_size is the percentage of events assigned to the test set (remainder go into training)
### feature matrices changed to dense representations for compatibility with classifier
### functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = (vectorizer.fit_transform(features_train)).toarray()
features_test  = (vectorizer.transform(features_test)).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150]
labels_train   = labels_train[:150]



### your code goes here
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train    )
pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
print('Accuracy: {}'.format(accuracy))

important_list = []
importances = clf.feature_importances_
for index, value in enumerate(importances):
    if value >= 0.2:
        important_list.append([index,value])

important_list.sort(key=lambda x: x[1])
important_index = important_list[0][0]

important_word = vectorizer.get_feature_names()[important_index]
#print("most important word: ", important_word)

for i in [x[0] for x in important_list]:
    print(vectorizer.get_feature_names()[i])


        