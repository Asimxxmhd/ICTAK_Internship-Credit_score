import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request, redirect, url_for

data = pd.read_csv('D:/Demo_flask/Internship_DSA/credit.csv')

Q1 = np.percentile(data['Total_EMI_per_month'],25,method='midpoint')
Q2 = np.percentile(data['Total_EMI_per_month'],50,method ='midpoint')
Q3 = np.percentile(data['Total_EMI_per_month'],75,method = 'midpoint')

IQR = Q3 - Q1
low_lim = Q1 - 1.5*IQR
up_lim = Q3 + 1.5*IQR

outlier = []
for x in data['Total_EMI_per_month']:
    if ((x>up_lim) or (x<low_lim)):
        outlier.append(x)

ind2 = data['Total_EMI_per_month']>up_lim
indes = data.loc[ind2].index
indes

data.drop(indes,inplace=True)

Q1 = np.percentile(data['Amount_invested_monthly'],25,method='midpoint')
Q2 = np.percentile(data['Amount_invested_monthly'],50,method='midpoint')
Q3 = np.percentile(data['Amount_invested_monthly'],75,method ='midpoint')

IQR = Q3 - Q1
low_lim = Q1 - 1.5*IQR
up_lim = Q3 + 1.5*IQR

outlier=[]
for x in data['Amount_invested_monthly']:
    if x<low_lim:
        outlier.append(x)


ind2 = data['Amount_invested_monthly']>up_lim
indes = data.loc[ind2].index

data.drop(indes,inplace=True)


Q1 = np.percentile(data['Monthly_Balance'],25,method='midpoint')
Q2 = np.percentile(data['Monthly_Balance'],50,method ='midpoint')
Q3 = np.percentile(data['Monthly_Balance'],75,method = 'midpoint')


ind_1 = data['Monthly_Balance']<low_lim
inds_1 = data.loc[ind_1].index
data.drop(inds_1,inplace=True)    

ind2 = data['Monthly_Balance']>up_lim
indes = data.loc[ind2].index

data.drop(indes,inplace=True)

data = data.drop('ID',axis=1)
data = data.drop('Customer_ID',axis=1)
data = data.drop('Name',axis=1)
data = data.drop('Age',axis=1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

cols = ['Occupation','Type_of_Loan','Credit_Mix','Payment_of_Min_Amount','Payment_Behaviour','Credit_Score']

for col in cols:
    data[col] = le.fit_transform(data[col])

X = data.drop(['Credit_Score'],axis=1)
y = data['Credit_Score']

from collections import Counter
from imblearn.over_sampling import RandomOverSampler

# Assuming x and y are features and target variable
X = data[['Payment_Behaviour','Payment_of_Min_Amount','Credit_Mix','Occupation','Num_Bank_Accounts','Num_Credit_Card','Interest_Rate','Num_of_Loan']]
y = data['Credit_Score']

ros = RandomOverSampler(random_state=42)

# Fit predictor and target variable
X_ros, y_ros = ros.fit_resample(X, y)

from sklearn.model_selection import train_test_split

X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(X_ros, y_ros, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.ensemble import RandomForestClassifier
rf_cls = RandomForestClassifier()
model_rf = rf_cls.fit(X_train_temp,y_train_temp)
y_pred_rf = model_rf.predict(X_val)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
acc = accuracy_score(y_val,y_pred_rf)
print(acc)

# Save the trained model using pickle
model_filename = 'credit_score.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(rf_cls, model_file)