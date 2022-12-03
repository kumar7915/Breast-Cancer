''' Breast-Cancer-Prediction '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

dk = pd.read_csv(r"C:\Users\psaik\Downloads\data.csv")
dk.info()
dk.describe()

corr=dk.corr()
sns.heatmap(corr)

dk.drop(['id'],axis=1,inplace=True)
dk.head()

dk.isna().sum()

import dtale
dtale.show(dk)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler    # CLassification- StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost

enc=OrdinalEncoder()
enc.fit(dk[['diagnosis']])
dk[['diagnosis']]=enc.transform(dk[['diagnosis']])

X=dk.drop(['diagnosis'],axis=1)
Y=dk['diagnosis']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=27)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# Building Pipeline
pipeline=Pipeline([
    #("Logistic Regression",LogisticRegression(random_state=42))
    #("Random Forest",RandomForestClassifier())
    ("Ada Boost Classifier",AdaBoostClassifier(RandomForestClassifier()))
])

pipeline.fit(X_train,Y_train)

from sklearn.metrics import mean_absolute_error
Y_pred = pipeline.predict(X_test)
print('Mean Absolute Error: ', mean_absolute_error(Y_pred, Y_test))
print('Score', pipeline.score(X_test, Y_test))

from sklearn.metrics import accuracy_score
accuracy_score(Y_pred,Y_test)
