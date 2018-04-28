from pandas import Series,DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
%matplotlib inline
from sklearn.cross_validation import train_test_split
import sklearn.metrics
import seaborn as sns
import os
os.chdir('c://users/sujith/desktop')
iris=pd.read_csv("iris.csv")
iris.head()
sns.pairplot(data=iris)
feature_cols=['sepal_length','petal_length','sepal_width','petal_width']
x=iris[feature_cols]
y=iris.species
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=25)
classifier.fit(x_train,y_train)
predictions=classifier.predict(x_test)
conf_matrix=sklearn.metrics.confusion_matrix(y_test,predictions)
conf_matrix
sklearn.metrics.accuracy_score(y_test,predictions)