import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# Classifiers
from sklearn.linear_model import Perceptron
from sklearn.linear_model.ridge import RidgeClassifierCV
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from mlxtend.classifier import StackingCVClassifier # <- Here is our boy

# Used to ignore warnings generated from StackingCVClassifier
import warnings
warnings.simplefilter('ignore')

rosemary = pd.read_csv('C:/Users/GALTUN/.spyder-py3/Class.csv')
rosemary.drop(['Class','Date','L1','XU030 Index - Volume'],1)
rosemary = rosemary.dropna(axis = 0)
#rosemary['XU030 Index - Volume'] = rosemary['XU030 Index - Volume'].astype(float)

#print(rosemary)

#corrmat = rosemary.corr()
#top_corr_features = corrmat.index
#plt.figure(figsize=(20,20))
#plot heat map
#g=sns.heatmap(rosemary[top_corr_features].corr(),annot=True,cmap="RdYlGn")

X = rosemary.drop(['Class','Date','L1','XU030 Index - Volume','USDTRY MA 20','SMAVG (50)','DXY(-3)','SMAVG (5)', 'Return in 180 days','SPX(-3)',
                   '10Yr(-1)','VIX(-3)','10Yr(-2)','Close(-4)','EEM(-3)', 'USDTRY Vol 10', 'Close (-5)', 'SMAVG (100)','SPX Vol 10',
                   'USDTRY(-1)','CL1 MA 5','Close (-1)','XAUXAG(-2)','CL1 MA 50','USDTRY MA 50'],1)
y = rosemary['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.70, test_size = 0.30)
mm = MinMaxScaler()
X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)

#initializing classifiers
classifier1 = MLPClassifier(activation = 'relu', solver = 'adam', hidden_layer_sizes =(64 , 64 , 32), batch_size = 50, learning_rate = 'invscaling', max_iter = 2500)
#classifier5 = RandomForestClassifier(n_estimators = 500, criterion = 'gini', max_depth = 10, max_features = 'auto', min_samples_leaf = 0.005, min_samples_split = 0.005, n_jobs = -1, random_state=1000)
classifier2 = LGBMClassifier()
#classifier3 = BernoulliNB()
classifier4 = CatBoostClassifier()
classifier5 = ExtraTreesClassifier(criterion = 'gini',bootstrap = True, oob_score = 'True', class_weight = 'balanced_subsample')

#Stacking
sclf = StackingCVClassifier(classifiers = [classifier1, classifier2, classifier4, classifier5],
                            shuffle = False,
                            use_probas = True,
                            cv = 5,
                            meta_classifier = SVC(probability = True))

#List to store classifiers
classifiers = {"MLP": classifier1,
               "LGBM": classifier2,
#               "NB": classifier3,
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
    
# Get results
results = pd.DataFrame()
for key in classifiers:
    # Make prediction on test set
    y_pred = classifiers[key].predict_proba(X_test)[:,1]
    
    # Save results in pandas dataframe object
    results[f"{key}"] = y_pred

# Add the test set to the results object
results["Target"] = y_test

# Probability Distributions Figure
# Set graph style
sns.set(font_scale = 1)
sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
               "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
               'ytick.color': '0.4'})

# Plot
f, ax = plt.subplots(figsize=(13, 4), nrows=1, ncols = 5)

for key, counter in zip(classifiers, range(5)):
    # Get predictions
    y_pred = results[key]
    
     # Get AUC
    auc = metrics.roc_auc_score(y_test, y_pred)
    textstr = f"AUC: {auc:.3f}"

        # Plot false distribution
    false_pred = results[results["Target"] == 0]
    sns.distplot(false_pred[key], hist=True, kde=False, 
                 bins=int(25), color = 'red',
                 hist_kws={'edgecolor':'black'}, ax = ax[counter])
    
    # Plot true distribution
    true_pred = results[results["Target"] == 1]
    sns.distplot(results[key], hist=True, kde=False, 
                 bins=int(25), color = 'green',
                 hist_kws={'edgecolor':'black'}, ax = ax[counter])
    
    
    # These are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    # Place a text box in upper left in axes coords
    ax[counter].text(0.05, 0.95, textstr, transform=ax[counter].transAxes, fontsize=14,
                    verticalalignment = "top", bbox=props)
    
    # Set axis limits and labels
    ax[counter].set_title(f"{key} Distribution")
    ax[counter].set_xlim(0,1)
    ax[counter].set_xlabel("Probability")
    
# Tight layout
plt.tight_layout()

# Save Figure
plt.savefig("Probability Distribution for each Classifier.png", dpi = 1080)


# Define parameter grid 
params = {"meta_classifier__kernel": ['rbf', 'poly'],
          "meta_classifier__C": [1, 2],
          "meta_classifier__degree": [3, 4, 5],
          "meta_classifier__probability": [True]}


# Initialize GridSearchCV
grid = GridSearchCV(estimator = sclf, 
                    param_grid = params, 
                    cv = 5,
                    scoring = "roc_auc",
                    verbose = 10,
                    n_jobs = -1)

# Fit GridSearchCV
grid.fit(X_train, y_train)

# Making prediction on test set
y_pred = grid.predict_proba(X_test)[:,1]

# Getting AUC
auc = metrics.roc_auc_score(y_test, y_pred)

# Print results
print(f"The AUC of the tuned Stacking classifier is {auc:.3f}")

    
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func = f_classif, k='all')
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  
print(featureScores.nlargest(10,'Score'))  