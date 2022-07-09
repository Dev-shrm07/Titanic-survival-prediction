import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv('titanic.csv')
data.head(10)
data.dropna(inplace=True)
data.drop("Cabin", axis = 1, inplace = True)
sex = pd.get_dummies(data['Sex'], drop_first = True)
sex.head(5)
pcl = pd.get_dummies(data['Pclass'], drop_first = True)
pcl.head(5)
data = pd.concat([data, pcl, sex, embarked], axis = 1)
data.head(5)
data.drop(['Embarked', 'Pclass', 'Name', 'PassengerId','Sex','Ticket'], axis = 1, inplace = True)
data.head(5)
X = data.drop(['Survived'], axis = 1)
y = data['Survived']
from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state = 1)
from sklearn.linear_model import LogisticRegression as lr
reg = lr()
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
