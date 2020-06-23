# -*- coding: utf-8 -*-
"""Spike.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_RjJej0wk6hOcjON4QcknRZz67OUM8VA
"""

from google.colab import drive
drive.mount('/content/drive/')

!pip install yahoofinancials

pip install pandas-ta

!pip install pycaret

!pip install costcla

import pandas as pd
from numpy import mean
from datetime import datetime
import matplotlib.pyplot as plt
from yahoofinancials import YahooFinancials
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
import numpy as np
import seaborn as sns
import scipy.stats as st
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from catboost import CatBoostClassifier
from catboost.utils import Pool, get_confusion_matrix
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from costcla.metrics import cost_loss, savings_score
from costcla.models import CostSensitiveRandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import plot_confusion_matrix
from yellowbrick.classifier import ConfusionMatrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier, plot_importance, DMatrix, plot_tree
import pandas_ta as ta

ticker_details = pd.read_excel('/content/drive/My Drive/Colab Notebooks/Ticker List.xlsx')

ticker = ticker_details['Ticker'].to_list()
names = ticker_details['Description'].to_list()

#Extracting Data from Yahoo Finance and Adding them to Values table using date as key
end_date= "2020-06-19"
start_date = "2000-01-01"
date_range = pd.bdate_range(start=start_date,end=end_date)
values = pd.DataFrame({ 'Date': date_range})
values['Date']= pd.to_datetime(values['Date'])

#Extracting Data from Yahoo Finance and Adding them to Values table using date as key
for i in ticker:
    raw_data = YahooFinancials(i)
    raw_data = raw_data.get_historical_price_data(start_date, end_date, "daily")
    df = pd.DataFrame(raw_data[i]['prices'])[['formatted_date','adjclose']]
    df.columns = ['Date1',i]
    df['Date1']= pd.to_datetime(df['Date1'])
    values = values.merge(df,how='left',left_on='Date',right_on='Date1')
    values = values.drop(labels='Date1',axis=1)

#Renaming columns to represent instrument names rather than their ticker codes for ease of readability
names.insert(0,'Date')
values.columns = names
#print(values.shape)
#print(values.isna().sum())
values.tail()

#Front filling the NaN values in the data set
values = values.fillna(method="ffill",axis=0)
values = values.fillna(method="bfill",axis=0)
values.isna().sum()

# Co-ercing numeric type to all columns except Date
cols=values.columns.drop('Date')
values[cols] = values[cols].apply(pd.to_numeric,errors='coerce').round(decimals=4)
#print(values.tail())

values['SPX-RSI'] = ta.rsi( values['SPX'] )

BBANDS = ta.bbands( values['SPX'] )
keys = BBANDS.keys().to_list()

Upper = BBANDS[ 'BBU_5' ]
Lower = BBANDS[ 'BBL_5' ]

Upper_perc = Upper / values['SPX']
Lower_perc = Lower / values['SPX']

values[ 'BBU-Distance' ] = Upper_perc
values[ 'BBL-Distance' ] = Lower_perc
values['MACD-Histogram'] = ta.macd( values[ 'SPX' ] )[ 'MACDH_12_26_9' ]

imp = ['Gold','USD Index', 'Oil', 'SPX','VIX', 'High Yield Fund' , 'Nikkei', 'Dax', '10Yr', '2Yr' , 'EEM' ,'XLE', 'XLF', 'XLI', 'AUDJPY']
# Calculating Short term -Historical Returns
change_days = [1,3,5,14,21]

data = pd.DataFrame(data=values['Date'])
for i in change_days:
    print(data.shape)
    x= values[cols].pct_change(periods=i).add_suffix("-T-"+str(i))
    data=pd.concat(objs=(data,x),axis=1)
    x=[]
#print(data.shape)

# Calculating Long term Historical Returns
change_days = [60,90,180,250]

for i in change_days:
    print(data.shape)
    x= values[imp].pct_change(periods=i).add_suffix("-T-"+str(i))
    data=pd.concat(objs=(data,x),axis=1)
    x=[]
#print(data.shape)

#Calculating Moving averages for SPX
moving_avg = pd.DataFrame(values['Date'],columns=['Date'])
moving_avg['Date']=pd.to_datetime(moving_avg['Date'],format='%Y-%b-%d')
moving_avg['SPX/15SMA'] = (values['SPX']/(values['SPX'].rolling(window=15).mean()))-1
moving_avg['SPX/30SMA'] = (values['SPX']/(values['SPX'].rolling(window=30).mean()))-1
moving_avg['SPX/60SMA'] = (values['SPX']/(values['SPX'].rolling(window=60).mean()))-1
moving_avg['SPX/90SMA'] = (values['SPX']/(values['SPX'].rolling(window=90).mean()))-1
moving_avg['SPX/180SMA'] = (values['SPX']/(values['SPX'].rolling(window=180).mean()))-1
moving_avg['SPX/90EMA'] = (values['SPX']/(values['SPX'].ewm(span=90,adjust=True,ignore_na=True).mean()))-1
moving_avg['SPX/180EMA'] = (values['SPX']/(values['SPX'].ewm(span=180,adjust=True,ignore_na=True).mean()))-1
moving_avg = moving_avg.dropna(axis=0)
#print(moving_avg.shape)
#print(moving_avg.head())

#Merging Moving Average values to the feature space
#print(data.shape)
data['Date']=pd.to_datetime(data['Date'],format='%Y-%b-%d')
data = pd.merge(left=data,right=moving_avg,how='left',on='Date')
#print(data.shape)
data.isna().sum()

#Caluculating forward returns for Target
y = pd.DataFrame(data=values['Date'])
print(y.shape)

y['SPX-T+14']=values["SPX"].pct_change(periods=-14)
y['SPX-T+22']=values["SPX"].pct_change(periods=-22)
print(y.shape)
y.isna().sum()
len(names)

# Removing NAs
print(data.shape)
data = data[data['SPX-T-250'].notna()]
y = y[y['SPX-T+22'].notna()]
print(data.shape)
print(y.shape)

#Adding Target Variables
data = pd.merge(left=data,right=y,how='inner',on='Date',suffixes=(False,False))
print(data.shape)
data.isna().sum()

sns.distplot(data['SPX-T+14'])
sns.distplot(data['SPX-T+22'])

#Select Threshold p (left tail probability)
p1 = 0.25
p2 = 0.75
#Get z-Value
z1 = st.norm.ppf(p1)
z2 = st.norm.ppf(p2)
print(z1)
print(z2)

aa = data.corr()

aa.keys()

#Calculating Threshold (t) for each Y
t_141 = round((z1*np.std(data["SPX-T+14"]))+np.mean(data["SPX-T+14"]),5)
t_142 = round((z2*np.std(data["SPX-T+14"]))+np.mean(data["SPX-T+14"]),5)
t_221 = round((z1*np.std(data["SPX-T+22"]))+np.mean(data["SPX-T+22"]),5)
t_222 = round((z2*np.std(data["SPX-T+22"]))+np.mean(data["SPX-T+22"]),5)

print("t_141=",t_141)
print("t_142=",t_142)
#print("t_221=",t_221)
#print("t_222=",t_222)

#Creating Labels
data['Y-141'] = (data['SPX-T+14']< t_141)*1
data['Y-142'] = (data['SPX-T+14']> t_142)*1
data['Y-221']= (data['SPX-T+22']< t_221)*1
data['Y-222']= (data['SPX-T+22']> t_222)*1
#print("Y-141", sum(data['Y-141']))
#print("Y-142", sum(data['Y-142']))
#print("Y-221", sum(data['Y-221']))
#print("Y-222", sum(data['Y-222']))

data_TEST = data.iloc[: -150]

data = data.drop(['SPX-T+14','SPX-T+22','Date'],axis=1)
#print(data.head())

data_14 = data.drop(['Y-221','Y-222','Y-141'],axis=1)
X = data_14.drop(['Y-142'],axis=1)
y= data_14['Y-142']
feature_names = list(X.columns.values)
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)

mm = MinMaxScaler()
X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)

smt = SMOTETomek()
X_train, y_train = smt.fit_sample(X_train, y_train)

class_weight = int(sum(y_train == 0)/sum(y_train==1))
class_weight

vif_scores = variance_inflation_factors( no_target_data )

inds = (vif_scores > 10 ).index
droplist = inds[vif_scores > 10 ]

if multicollinearity == True:
    data = data.drop( droplist, axis = 1 )



def variance_inflation_factors( exog_df ):

    exog_df = add_constant(exog_df)
    vifs = pd.Series(
        [1 / (1. - OLS(exog_df[col].values, 
                      exog_df.loc[:, exog_df.columns != col].values).fit().rsquared) 
        for col in exog_df],
        index=exog_df.columns,
        name='VIF'
    )
    return vifs

knn = KNeighborsClassifier(n_neighbors = 1, weights = 'distance', leaf_size= 42,p = 2)
knn.fit(X_train,y_train)
y_pred = knn.predict_proba(X_test)
cm = ConfusionMatrix(knn)
cm.fit(X_train, y_train)
cm.score(X_test, y_test)
cm.show()

k_range = np.arange(1,61)
weights = ["uniform","distance"]
p = [1,2]
leaf_size_range = np.arange(1,71)
param_grid = dict(n_neighbors = k_range, weights = weights, leaf_size = leaf_size_range,p = p)
knn = KNeighborsClassifier()
randomized = RandomizedSearchCV(knn, param_grid,scoring = "precision", cv = 5, n_iter = 20)
randomized.fit(X_train,y_train)
randomized.best_estimator_

catb = CatBoostClassifier(iterations=5000,scale_pos_weight=6,12)
catb.fit(X_train, y_train)
y_pred = catb.predict_proba(X_test)
cm = get_confusion_matrix(catb, Pool(X_test, y_test))
print(cm)

iterations = np.arange(1,5000)
leaf_reg = np.arange(2,30)
param_grid = dict(iterations = iterations, 12_leaf_reg = leaf_reg )
catb = CatBoostClassifier()
randomized = RandomizedSearchCV(catb, param_grid,scoring = "precision", cv = 5, n_iter = 20)
randomized.fit(X_train,y_train)
randomized.best_estimator_

et = ExtraTreesClassifier(class_weight= "balanced", max_features= "log2", n_estimators = 325)
et.fit(X_train,y_train)
y_pred = et.predict_proba(X_test)
cm = ConfusionMatrix(et)
cm.fit(X_train, y_train)
cm.score(X_test, y_test)
cm.show()

estimator_range = np.arange(1,350)
criterion = ["gini","entropy"]
depth = np.arange(1,100)
features = ["auto","sqrt","log2"]
class_weight = ["balanced", "balanced_subsample"]
leaf_size_range = np.arange(1,71)
param_grid = dict(n_estimators = estimator_range, criterion = criterion, max_features = features, class_weight = class_weight)
et = ExtraTreesClassifier()
randomized = RandomizedSearchCV(et, param_grid,scoring = "precision", cv = 5, n_iter = 20)
randomized.fit(X_train,y_train)
randomized.best_estimator_

lgbm = LGBMClassifier(class_weight={0: 6, 1: 1})
lgbm.fit(X_train,y_train)
y_pred = lgbm.predict_proba(X_test)
cm = ConfusionMatrix(lgbm)
cm.fit(X_train, y_train)
cm.score(X_test, y_test)
cm.show()

xgb = XGBClassifier( scale_pos_weight = 6,base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=5,
              min_child_weight=1, missing=None, n_estimators=200, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, seed=123,
              silent=None, subsample=1, verbosity=1)
xgb_gs.fit(X_train,y_train)
print ("_________________________________Model Selection_____________________________________")
print ("Best estimator found by grid search:",xgb_gs.best_estimator_)
print ("Best parameters found by grid search:",xgb_gs.best_params_)
print ("Best accuracy score found by grid search:",xgb_gs.best_score_)
print ()

def classifier_measures(y_test,model_pred):
    # output confusion matrix
    print ("-------------------------------------Accuracy-----------------------------------------")
    Accuracy = accuracy_score(y_test, model_pred)
    print ("Accuracy:",Accuracy)
    print ("---------------------------------Confusion Matrix-------------------------------------")
    model_matrix = confusion_matrix(y_test, model_pred)
    # print ('Confusion matrix:\n',model_matrix)
    fig= plt.figure(figsize=(6,3))# to plot the graph
    print("TP",model_matrix[1,1,]) # no of Fraud transaction which are predicted Fraud
    print("TN",model_matrix[0,0]) # no. of normal transaction which are predited normal
    print("FP",model_matrix[0,1]) # no of normal transaction which are predicted Fraud
    print("FN",model_matrix[1,0]) # no of Fraud Transaction which are predicted normal
    sns.heatmap(model_matrix,cmap="coolwarm_r",annot=True,linewidths=0.5)
    plt.title("Confusion_matrix")
    plt.xlabel("Predicted_class")
    plt.ylabel("Real class")
    plt.show()

    print()
    print ("--------------------------------Precession and Recall---------------------------------")
    target_names = ['0',"1"]
    print(classification_report(y_test, model_pred, target_names=target_names))

# Predict new values using best model derived from cross validation and test performance on the test set
xgb_pred = xgb_gs.predict(X_test)
xgb_prob = xgb_gs.predict_proba(X_test)
xgb_score = xgb_gs.score(X_test, y_test)
print ("________________________________Model Prediction_____________________________________")
print ('F1-measure:', xgb_score)
print ()
classifier_measures(y_test,xgb_pred)



classifier1 = KNeighborsClassifier(n_neighbors = 4,weights= 'distance',p = 1)
classifier2 = LGBMClassifier(class_weight={0: 1, 1: 5})
classifier3 = XGBClassifier( scale_pos_weight = 6,base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=5,
              min_child_weight=1, missing=None, n_estimators=200, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, seed=123,
              silent=None, subsample=1, verbosity=1)
classifier4 = CatBoostClassifier(iterations=5000)
classifier5 = ExtraTreesClassifier(n_estimators=150,class_weight= {0: 5, 1: 1})

sclf = StackingCVClassifier(classifiers = [classifier1, classifier2, classifier4, classifier5],
                            shuffle = False,
                            use_probas = True,
                            cv = 5,
                            meta_classifier = CatBoostClassifier())

classifiers = {"KNN": classifier1,
               "LGBM": classifier2,
               "XGB": classifier3,
               "CatBoost": classifier4,
               "ET": classifier5,
               "Stack": sclf}

# Train classifiers
for key in classifiers:
    # Get classifier
    classifier = classifiers[key]
    
    # Fit classifier
    classifier.fit(X_train, y_train)
        
    # Save fitted classifier
    classifiers[key] = classifier

pred = sclf.predict( X_test)

# Get results
results = pd.DataFrame()
for key in classifiers:
    # Make prediction on test set
    y_pred = classifiers[key].predict(X_test)
    
    # Save results in pandas dataframe object
    results[f"{key}"] = y_pred

# Add the test set to the results object
results["Target"] = y_test

pred_stack = results[ 'Stack' ] 
score = f1_score( y_test, pred_stack ) 
score

from sklearn.metrics import confusion_matrix

pred = sclf.predict( X_test ) 

confusion_matrix( y_test, pred )

#DONT RUN 
#DONT RUN

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#mm = MinMaxScaler()
#X_train = mm.fit_transform(X_train)
#X_test = mm.transform(X_test)

#smt = SMOTETomek()
#X_train_SMOTE, y_train_SMOTE = smt.fit_sample(X_train, y_train)

#data_train = pd.concat( [ pd.DataFrame( X_train ), pd.DataFrame( y_train ) ], axis = 1 )
#data_train.columns = data_14.columns
#data_test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)],axis = 1)
#data_test.columns = data_14.columns
#data_train.head()
#data_test.head()

plot_confusion_matrix(sclf,X_test,y_test)
plt.show()

catb = create_model('catboost')

interpret_model(catb, plot='summary')

catb_tuned = tune_model('catboost', optimize = 'F1')

catb_boosted = ensemble_model(estimator = catb, method = 'Bagging')

knn = create_model('knn')

plot_model(knn, plot = 'confusion_matrix')

knn_tuned = tune_model('knn', n_iter = 100, optimize='F1')

plot_model(knn_tuned, plot = 'confusion_matrix')

lgbm = create_model('lightgbm')

plot_model(lgbm, plot = 'confusion_matrix')

lgbm_tuned = tune_model('lightgbm', optimize='F1')

plot_model(lgbm_tuned, plot = 'confusion_matrix')

stack1 = stack_models(estimator_list = [et,catb,lgbm_tuned,knn_tuned], meta_model = catb)

stack1_final = finalize_model(stack1)

save_model(stack1,'stack1')

pred_holdout = predict_model(stack1_final, data=data_test)