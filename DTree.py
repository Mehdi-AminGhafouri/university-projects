# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 22:01:12 2020

@author: Mehdi A.Ghafouri
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree

col_names = ['Result', 'Cylinder', 'Portable weight', 'Power', 'Weight', 'Acceleration', 'Year', 'country']
# load dataset
data = pd.read_csv("TreeData.csv", header=None, names=col_names)

feature_cols = ['Cylinder', 'Portable weight', 'Power', 'Weight', 'Acceleration', 'Year']
X = data[feature_cols] # Features
y = data.Result # Target variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

#Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

tree.plot_tree(clf)

from sklearn.tree import export_graphviz
import graphviz 
dot_data = export_graphviz(clf,out_file = None,filled = True, rounded = True,special_characters = True,feature_names=feature_cols,class_names=data.Result)
graph2 = graphviz.Source(dot_data)
graph2.render("MyTreePDF") 
