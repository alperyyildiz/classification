import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from lightgbm import LGBMClassifier

# Classifiers
from sklearn.gaussian_process import GaussianProcessClassifier
from catboost import CatBoostClassifier
from sklearn.svm import NuSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier # <- Here is our boy :):)

# Used to ignore warnings generated from StackingCVClassifier
import warnings
warnings.simplefilter('ignore')

rosemary = pd.read_excel('SON.xlsx')
rosemary = rosemary.dropna( axis = 0)


#Create Feature/Class sets
X = rosemary.drop(['Class','Date','L1','XU030 Index - Volume','SPX Vol 20','VIX(-3)','Month of the Year','CL1 MA 50','CL1 MA 20','SPX Vol 50','SMAVG (21)','10Yr','SPX(-3)','CL1 MA 5',
                   'DAX(-3)','Day of the Week','SMAVG (5)','10Yr(-1)','10Yr(-3)','SMAVG (200)','CL1 (-3)','NIKKEI','DXY(-1)','CL1','USDTRY(-2)'
                   ,'SMAVG (100)','SMAVG (13)','CL1 MA 10','SPX(-2)','NIKKEI(-2)','Close(-4)','SMAVG (50)','10Yr(-2)','SPX MA 50',
                   'EEM(-3)','VIX(-2)','VIX(-1)','CL1 (-1)'],1)
y = rosemary['Class']



  

BASE = 487
cycle = 5 * 2                #Num of days for model to be trained again
loop_range = BASE // cycle   #Num of training needed


#Initialize collection of results
LAST_RESULTS = pd.DataFrame()


for PART in range( loop_range ):
    if PART == loop_range - 1:
        ITER_DAYS = PART * cycle

        #Create Train/Test sets
        X_train = X.iloc[ : -BASE + ITER_DAYS ]
        y_train = y.iloc[ : -BASE + ITER_DAYS ]

        X_test = X.iloc[ -BASE + ITER_DAYS : ]
        y_test = y.iloc[ -BASE + ITER_DAYS : ]

    else:
        ITER_DAYS = PART * cycle

        #Create Train/Test sets
        X_train = X.iloc[ : -BASE + ITER_DAYS ]
        y_train = y.iloc[ : -BASE + ITER_DAYS ]

        X_test = X.iloc[ -BASE + ITER_DAYS : -BASE + ITER_DAYS + cycle ]
        y_test = y.iloc[ -BASE + ITER_DAYS : -BASE + ITER_DAYS + cycle ]

    print(PART)
    print('\n')
    print(y_train.shape)
    print('\n')
    print(y_test.shape)
    print('\n')


    mm = MinMaxScaler()
    X_train = mm.fit_transform(X_train)
    X_test = mm.transform(X_test)


    #initializing classifiers
    classifier1 = MLPClassifier(activation = 'relu', solver = 'adam', alpha = 0.1, hidden_layer_sizes = (16, 16, 16), learning_rate = 'constant', max_iter = 2000, random_state=1000)
    classifier2 = RandomForestClassifier(n_estimators = 500, criterion = 'gini', max_depth = 15, max_features = 'auto', min_samples_leaf = 0.005, min_samples_split = 0.005, n_jobs = -1, random_state=1000)
    classifier3 = SVC(C = 50, degree = 1, gamma = "auto", kernel = "rbf", probability = True)
    classifier4 = CatBoostClassifier()
    classifier5 = LGBMClassifier()

    #Stacking
    sclf = StackingCVClassifier(classifiers = [classifier1, classifier2,classifier3, classifier4],
                                shuffle = False,
                                use_probas = True,
                                cv = 5,
                                meta_classifier = CatBoostClassifier())

    #List to store classifiers
    classifiers = {"SVC": classifier3,
                  "MLP": classifier1,
                  "CatBoost": classifier4,
                  "RF": classifier2,
                  "LGBM": classifier5,
                  "Stack": sclf}
    
    TEMP_RESULTS = pd.DataFrame()  
    # Train classifiers
    for key in classifiers:
        # Get classifier
        classifier = classifiers[ key ]
        
        # Fit classifier
        classifier.fit( X_train, y_train )
            
        # Save fitted classifier
        classifiers[key] = classifier

        # Make prediction on test set
        y_pred = classifiers[ key ].predict_proba( X_test )[ : , 1 ]

        # Save results in TEMP_RESULTS
        TEMP_RESULTS[f"{key}"] = y_pred
    
    #Append results in LAST_RESULTS
    LAST_RESULTS = LAST_RESULTS.append( TEMP_RESULTS )
    



# Add the test set to the results object
y_test = y.iloc[ -BASE : ]
#CHANGED HERE since it returns NaN in dataframe
LAST_RESULTS["Target"] = np.array( y_test )
 

# Probability Distributions Figure
# Set graph style
sns.set(font_scale = 1)
sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
               "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
               'ytick.color': '0.4'})

# Plot
f, ax = plt.subplots(figsize=(13, 4), nrows=1, ncols = 5)

for key, counter in zip( classifiers, range( 5 ) ):
    # Get predictions
    y_pred = LAST_RESULTS[key]
    
     # Get AUC
    auc = metrics.roc_auc_score( y_test, y_pred )
    textstr = f"AUC: {auc:.3f}"

    # Plot false distribution
    false_pred = LAST_RESULTS[ LAST_RESULTS[ "Target" ] == 0 ]
    sns.distplot( false_pred[ key ], hist = True, kde = False, 
                 bins = int(25), color = 'red',
                 hist_kws = { 'edgecolor':'black'}, ax = ax[ counter ] )
    # Plot true distribution
    true_pred = LAST_RESULTS[ LAST_RESULTS[ "Target" ] == 1 ]
    
    #CHANGED HERE results to true_pred in below line
    sns.distplot(true_pred[ key ], hist=True, kde=False, 
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

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func = f_classif, k='all')
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  
print(featureScores.nsmallest(10,'Score'))  
