# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 21:10:16 2018

@author: Aditya Narayanan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
#%%
clinical_data=pd.read_csv('../clinical_data.csv')
bill_amt=pd.read_csv('../bill_amount.csv')
bill_id=pd.read_csv('../bill_id.csv')
demographics=pd.read_csv('../demographics.csv')
#%% join data
df=bill_id.merge(bill_amt, on=['bill_id'], how='right')
clinical_data=clinical_data.rename(columns={'id':'patient_id'})
df=df.merge(clinical_data, how='left', on=['patient_id','date_of_admission'])
#%% clean data to remove bad spellings
demographics['gender']=demographics['gender'].apply(lambda x: 'M' if re.match(x,'m|M') else 'F')
demographics['resident_status']=demographics['resident_status'].apply(lambda x: 'Singaporean' if x.find('citizen')>=0 else x)
demographics['race']=demographics['race'].apply(lambda x: re.sub("Indiann$","Indian","%s%s" % (x[0].upper(), x[1:])))
df=df.merge(demographics, how='left', on='patient_id')
#%% check na values and replace
df.isnull().sum()
df['medical_history_3']=df['medical_history_3'].apply(lambda x: 1 if re.match(x,'1|Y') else 0)
df['medical_history_5']=df['medical_history_5'].fillna(3)
df['medical_history_2']=df['medical_history_2'].fillna(3)
#%% Convert date times to meaningful variables
df['date_of_admission']=pd.to_datetime(df['date_of_admission'], format='%Y-%m-%d')
df['date_of_discharge']=pd.to_datetime(df['date_of_discharge'], format='%Y-%m-%d')
df['date_of_birth']=pd.to_datetime(df['date_of_birth'], format='%Y-%m-%d')
df['Age']=(df['date_of_admission']-df['date_of_birth'])
df['Age']=df['Age'].apply(lambda x: x.days/365.25)
#%%Study response variable
df['amount'].describe()
f, ax = plt.subplots(figsize=(8,8))
sns.distplot(df['amount'], kde=False)
plt.show()
df['log_amount']=np.log(df['amount'])
f, ax = plt.subplots(figsize=(8,8))
sns.distplot(df['log_amount'], kde=Fsalse)
plt.show()
#%% Composite features
df['admit_time']=(df['date_of_discharge']-df['date_of_admission'])
df['admit_time']=df['admit_time'].apply(lambda x: x.days)
df['bmi']=df['weight']/(df['height']*df['height'])*10000
df['symptom_sum']=df['symptom_1']+df['symptom_2']+df['symptom_4']+df['symptom_4']+df['symptom_5']
df['preop_medication_sum']=df['preop_medication_1']+df['preop_medication_2']+df['preop_medication_3']+df['preop_medication_4']+df['preop_medication_5']+df['preop_medication_6']
df=df.drop(['bill_id','date_of_admission','date_of_discharge','date_of_birth','patient_id'],axis=1)
#%% plot univariate correlations
sns.pairplot(df, x_vars=["lab_result_1", "lab_result_2", "lab_result_3", "weight", "height", "bmi", "symptom_sum", "preop_medication_sum", "Age"], y_vars=["log_amount"])
plt.savefig('../myfig.png')
#%% plot box plots for categorical variables
ax = sns.boxplot(y="log_amount", x="resident_status", data=df)
#%%
df=df.join(pd.get_dummies(df[['gender','race','resident_status']]))
df=df.drop(['gender','race','resident_status','medical_history_5','medical_history_2'],axis=1)
#%%split trainigng and testing dataset
df=df.drop(labels='amount', axis=1)
msk=np.random.rand(len(df))< 0.8
x_train=df[msk].astype('float64')
y_train=x_train['log_amount'].astype('float64')
x_train=x_train.drop(labels='log_amount', axis=1).astype('float64')
x_test=df[~msk].astype('float64')
y_test=x_test['log_amount'].astype('float64')
x_test=x_test.drop(labels='log_amount', axis=1).astype('float64')
#%%
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
clf=SelectKBest(f_regression, k=10)
X_new1 = clf.fit_transform(X=np.asarray(x_train.values,dtype="float64"), y=(np.asarray(y_train, dtype="float64")))
mask1 = clf.get_support()
new_features = x_train.columns[mask1]
print(new_features)
#%%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
clf = RandomForestRegressor (n_estimators=100, max_features='sqrt', n_jobs=-1, random_state=0)
X_new = clf.fit(X=np.asarray(x_train.values,dtype="float64"), y=(np.asarray(y_train, dtype="float64")))
importances = clf.feature_importances_
indices = np.argsort(importances)
plY=x_train.columns[indices]
plX=importances[indices]
f, ax = plt.subplots(figsize=(15,15))
sns.barplot(x=plX, y=plY)
plt.show()
#%%