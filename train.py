#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mutual_info_score
from sklearn.pipeline import make_pipeline

#get_ipython().system('pip install tqdm')
#from tqdm.auto import tqdm

import pickle

# Read the data from the csv file into a dataframe
df = pd.read_csv('diabetes_prediction_dataset.csv')

categorical = list(df.dtypes[df.dtypes=='str'].index)
categorical

numerical = list(df.dtypes[df.dtypes!='str'].index)
numerical

# First remove the spaces in 'No Info' and 'not current'
df['smoking_history'] = df['smoking_history'].str.replace('No Info', 'No_Info')
df['smoking_history'] = df['smoking_history'].str.replace('not current', 'not_current')
df['smoking_history'].head()

# Continue editing the values of the categorical to remove the remaining extra spaces
for value in categorical:
    df[value] = df[value].str.lower().str.replace(' ','')

# We can examine the numerical columns (i.e. features)
#df.describe()

# Update the new numerical variables to exclude the target variable, diabetes
numerical = numerical[:-1]
numerical

# Try to normalize the distribution if not normalized
df.diabetes.value_counts(normalize=True)

# Proportion of positive diabetes values. The dataset is imbalanced with more non-diabetic cases
df.diabetes.mean()

# Split the data into train, validation and test sets (60%, 20%, 20%)
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_train, df_val = train_test_split(df_full_train, test_size=0.25, shuffle=False)

len(df_train) + len(df_val) + len(df_test) == len(df)

# Retreive the values of the target variable
y_full_train = df_full_train.diabetes.values
y_train = df_train.diabetes.values
y_val = df_val.diabetes.values
y_test = df_test.diabetes.values

len(y_train) + len(y_val) + len(y_test) == len(df)

# Compute feature importance (mutual information for categorical features)
def mutual_info_diabetes_score(series):
    return mutual_info_score(series, df_full_train.diabetes)

mi = df_full_train[categorical+numerical].apply(mutual_info_diabetes_score)

mi.sort_values(ascending=False)
mi

# Compute feature importance (correlation information for numerical features)
# Correlation matrix for the numerial variables
full_numerical = list(df.dtypes[df.dtypes!='str'].index)
corr_matrix = np.zeros([len(full_numerical),len(full_numerical)])
i = 0
for col in full_numerical:
    corr_matrix[i,:] = df_full_train[full_numerical].corrwith(df_full_train[col]).abs()
    i = i+1
corr_matrix.round(3)


# Compute the correlation matrix to see the dependence of the numberical features on the target variable
df_full_train[numerical].corrwith(df_full_train['diabetes']).abs().sort_values(ascending=False)

# Delete the target variable from the dataframes
#del df_full_train['loan_status']
del df_train['diabetes']
del df_val['diabetes']
del df_test['diabetes']

# Training the model
# One-hot encoding
dv = DictVectorizer(sparse=False)

train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)
X_train.shape

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)
X_val.shape

test_dict = df_test.to_dict(orient='records')
X_test = dv.transform(test_dict)
X_test.shape

# RMSE
def rmse(y_val, y_pred):
    square_error = (y_val - y_pred)**2
    MSE = np.mean(square_error)
    return np.sqrt(MSE)


del df_full_train['diabetes']

# Final model
# Best model is the Decision Tree classifier with max_depth=10, resulting in roc_auc_score=0.974 and rmse=0.152

full_train_dict = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(full_train_dict)

test_dict = df_test.to_dict(orient='records')
X_test = dv.transform(test_dict)

pipeline = make_pipeline(
    DictVectorizer(),
    DecisionTreeClassifier(max_depth = 10, random_state=1)
)

pipeline.fit(full_train_dict, y_full_train)

y_pred = pipeline.predict_proba(test_dict)[:,1]

auc = roc_auc_score(y_test, y_pred)

rmse(y_test, y_pred)

# Saving the model
with open('model.bin', 'wb') as f_out:
    pickle.dump(pipeline, f_out)

# Restart Kernel and run code from here to test that the model saved correctly and can do correct predictions
import pickle

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)
