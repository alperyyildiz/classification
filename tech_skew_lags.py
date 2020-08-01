import pandas as pd
from numpy import mean
from datetime import datetime
import matplotlib.pyplot as plt
from yahoofinancials import YahooFinancials
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns
import scipy.stats as st
import arch
from pytrends.request import TrendReq
from xgboost import XGBClassifier
from collections import OrderedDict 
from pytrends.request import TrendReq
from pytrends import dailydata
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import itertools
from pycaret.classification import *
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, roc_auc_score, confusion_matrix
from catboost import CatBoostRegressor
from lightgbm.sklearn import LGBMClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor    
from mlxtend.classifier import StackingCVClassifier # <- Here is our boy
import statsmodels.api as sm
import pandas_ta as ta

class HPOpt:

    def __init__(self, rosemary, space ):
        self.X_train = rosemary.X_train
        self.X_test  = rosemary.X_test
        self.y_train = rosemary.y_train
        self.y_test  = rosemary.y_test
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split( self.X_train, self.y_train, test_size = 0.15, random_state = 42, stratify = self.y_train )
        self.improved = list()
        self.int_keys = [ 'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 
                          'max_leaf_nodes', 'n_neighbors', 'leaf_size', 
                          'num_leaves', 'max_depth', 'n_estimators' ]



    def process(self, fn_name, space, trials, algo, max_evals):
      
        fn = getattr(self, fn_name)
        result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        return result, trials

    def find_best_threshold( self, preds ):
        best = 0.5
        best_score = 0
        for i in range( 150, 500 ):
            threshold = i / 1000
            new_preds = list()
            for k in range( len( preds ) ):
                if preds[ k ] > threshold:
                    new_preds.append( 1 )
                else:
                    new_preds.append( 0 )
            score = f1_score( self.y_val, new_preds )
            if score > best_score:
                best = threshold
                best_preds = new_preds
                best_score = score
        return best, best_preds
        
        

    def objective( self, args ):

        args_ = self.input_converter( args )
        print(args_)
        try:
            CLASSIFIER = ExtraTreesClassifier( random_state = 42, **args_ )
        except:
            CLASSIFIER = ExtraTreesClassifier( **args_ )

        CLASSIFIER.fit( self.X_train, self.y_train )

        nt_preds = CLASSIFIER.predict( self.X_test )
        nt_score = f1_score( self.y_test, nt_preds )

        pred_ = CLASSIFIER.predict_proba( self.X_val )[ :, 1 ]
        best, test_preds = self.find_best_threshold( pred_ )

        preds = CLASSIFIER.predict_proba( self.X_test )[ :, 1 ]
        test_preds = list()
        for k in range( len( preds ) ):
            if preds[ k ] > best:
                test_preds.append( 1 )
            else:
                test_preds.append( 0 )

        score = f1_score( self.y_test, test_preds )
        self.improved.append( ( nt_score, score, best ) )
        print( '\n ============================ \n {} \n ============================ \n'.format( nt_score ) )
        print( '\n ============================ \n {} \n ============================ \n'.format( score ) )
        print( '\n ============================ \n {} \n ============================ \n'.format( best ) )
        print( '\n {} \n'.format( args_ ) )
        cm = np.array( confusion_matrix( self.y_test, test_preds ) )
        plot_confusion_matrix( cm = cm, target_names = [ 'nothing', 'spike' ] )

        return {'loss': -score, 'status': STATUS_OK}

    def input_converter( self, args ):
        keyz = list( args.keys() )
        for key in keyz:
            if key in self.int_keys:
                args[ key ] = int( args[ key ] )
        return args 



# THIS ONE FOR CONFUSION MATRIX
def plot_confusion_matrix(cm,
                          target_names, 
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()



#PLOT HP AFFECTS AND CHANGE OVER TIME
def plot_param_vs_time( param_name ):
    f, ax = plt.subplots(1)
    xs = [t['tid'] for t in trials_obj.trials]
    ys = [t['misc']['vals'][ param_name ] for t in trials_obj.trials]
    ax.set_xlim(xs[0]-10, xs[-1]+10)
    ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
    ax.set_title('$' + param_name +'$ $vs$ $t$ ', fontsize=18)
    ax.set_xlabel('$t$', fontsize=16)
    ax.set_ylabel('$' + param_name +'$', fontsize=16)
    plt.show()

def plot_loss_vs_param( param_name ):
    f, ax = plt.subplots(1)
    xs = [t['misc']['vals'][ param_name ] for t in trials_obj.trials]
    ys = [t['result']['loss'] for t in trials_obj.trials]
    ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
    ax.set_title('$val$ $vs$ $' + param_name + '$ ', fontsize=18)
    ax.set_xlabel('$' + param_name + '$', fontsize=16)
    ax.set_ylabel('$val$', fontsize=16)
    plt.show()




class Data():
    def __init__( self ):
        super().__init__()
        self.ticker_details = pd.read_excel('Ticker List.xlsx')
        self.DF = pd.DataFrame()
        self.features_for_returns = ['Gold','USD Index', 'Oil', 'SPX','VIX', 'High Yield Fund' , 'Nikkei', 'Dax',
                                     '10Yr','2Yr' , 'EEM' ,'XLE', 'XLF', 'XLI', 'AUDJPY', 'XLK',
                                     'SSE', 'XLP','XLY', 'XLU', 'XLV', 'Lockheed', 'Lumber', 'Copper']

        self.the_list =    ['crash',
                            'spike', 'USD Index-T-1', 
                            'VIX-T-1', 'Gold-T-1', 'Oil-T-1',
                            'Nikkei-T-1', 'Dax-T-1', '10Yr-T-1',
                            '2Yr-T-1', 'EEM-T-1', 
                            'High Yield Fund-T-1', 'XLE-T-1',
                            'XLF-T-1', 'XLI-T-1', 'AUDJPY-T-1',
                            'XLK-T-1', 'SSE-T-1', 'XLP-T-1',
                            'XLY-T-1', 'XLU-T-1', 'XLV-T-1',
                            'Lumber-T-1', 'Lockheed-T-1', 
                            'Copper-T-1', 'USD Index-T-3',
                            'VIX-T-3', 'Gold-T-3', 'Oil-T-3',
                            'Nikkei-T-3', 'Dax-T-3', '10Yr-T-3',
                            '2Yr-T-3', 'EEM-T-3', 'High Yield Fund-T-3', 
                            'XLE-T-3', 'XLF-T-3', 'AUDJPY-T-3', 'XLK-T-3', 
                            'SSE-T-3', 'XLP-T-3', 'XLU-T-3', 'XLV-T-3',
                            'Lumber-T-3', 'Lockheed-T-3', 'Copper-T-3',
                            'USD Index-T-5', 'VIX-T-5', 'Gold-T-5', 
                            'Oil-T-5', 'Nikkei-T-5', 'Dax-T-5', '10Yr-T-5',
                            '2Yr-T-5', 'EEM-T-5', 'High Yield Fund-T-5', 
                            'XLE-T-5', 'XLF-T-5', 'AUDJPY-T-5', 'XLK-T-5', 
                            'SSE-T-5', 'XLP-T-5', 'XLY-T-5', 'XLU-T-5',
                            'XLV-T-5', 'Lumber-T-5', 'Lockheed-T-5', 
                            'Copper-T-5', 'USD Index-T-14', 'VIX-T-14',
                            'Gold-T-14', 'Oil-T-14', 'Nikkei-T-14', 
                            'Dax-T-14', '10Yr-T-14', '2Yr-T-14', 
                            'EEM-T-14', 'High Yield Fund-T-14',
                            'XLE-T-14', 'XLF-T-14', 'XLI-T-14', 
                            'AUDJPY-T-14', 'XLK-T-14', 'SSE-T-14',
                            'XLP-T-14', 'XLU-T-14', 'XLV-T-14',
                            'Lumber-T-14', 'Lockheed-T-14', 
                            'Copper-T-14', 'USD Index-T-21', 'VIX-T-21',
                            'Gold-T-21', 'Oil-T-21', 'Nikkei-T-21',
                            '10Yr-T-21', '2Yr-T-21', 'High Yield Fund-T-21',
                            'XLE-T-21', 'XLF-T-21', 'AUDJPY-T-21', 'XLK-T-21',
                            'SSE-T-21', 'XLP-T-21', 'XLU-T-21', 'XLV-T-21',
                            'Lumber-T-21', 'Lockheed-T-21', 'Copper-T-21', 
                            'SPX/vol5', 'SPX/vol180', 'Gold-T-60',
                            'USD Index-T-60', 'Oil-T-60', 'Nikkei-T-60',
                            'Dax-T-60', '10Yr-T-60', 'EEM-T-60',
                            'AUDJPY-T-60', 'XLK-T-60', 'SSE-T-60', 
                            'XLP-T-60', 'XLY-T-60', 'XLU-T-60',
                            'XLV-T-60', 'Lockheed-T-60', 'Lumber-T-60', 
                            'Copper-T-60', 'Gold-T-90', 'USD Index-T-90', 
                            'Oil-T-90', 'High Yield Fund-T-90', '10Yr-T-90',
                            'XLF-T-90', 'AUDJPY-T-90', 'SSE-T-90', 'XLU-T-90',
                            'XLV-T-90', 'Lockheed-T-90', 'Lumber-T-90',
                            'Gold-T-180', 'Oil-T-180', 'VIX-T-180', 
                            '10Yr-T-180', 'AUDJPY-T-180', 'XLU-T-180', 
                            'XLV-T-180', 'Lumber-T-180', 'USD Index-T-250',
                            'Oil-T-250', 'VIX-T-250', 'Nikkei-T-250',
                            '10Yr-T-250', '2Yr-T-250', 'XLE-T-250', 
                            'XLF-T-250', 'XLK-T-250', 'SSE-T-250',
                            'XLY-T-250', 'Lockheed-T-250', 
                            'Lumber-T-250', 'Copper-T-250']

                            
    def execute_all( self, set_date_inputs, st_ret_inputs,
                     volatility_inputs, lt_ret_inputs, 
                     ma_inputs, create_labels_inputs,
                     low = True, high = True, kurt = True ):
        self.set_date_range( **set_date_inputs )
        self.read_ticker_to_df()
        self.add_technical_ind()
        self.fill_na()
        self.add_st_returns( **st_ret_inputs )
        self.add_volatility(  **volatility_inputs )
        #self.add_garch()
        self.add_lt_returns( **lt_ret_inputs )
        self.add_moving_average( **ma_inputs )
        self.add_regime_probs(low = low, high = high)
        self.add_kurt()
        self.add_skew()
        self.add_technical_ind()
        self.create_labels( **create_labels_inputs )

    def set_date_range( self, start_date = "2000-01-01", end_date = "2020-07-31" ):
        self.start_date = start_date
        self.end_date = end_date
        self.DF = pd.DataFrame()
        #date_range = pd.bdate_range(start=start_date,end=end_date)
        #values = pd.DataFrame({ 'Date': date_range})
        #self.DF['Date']= pd.to_datetime(values['Date'])

    def read_ticker_to_df( self ):
        ticker = self.ticker_details['Ticker'].to_list()
        names = self.ticker_details['Description'].to_list()

        #Extracting Data from Yahoo Finance and Adding them to Values table using date as key
        for k, i in enumerate(ticker):
            raw_data = YahooFinancials(i)
            raw_data = raw_data.get_historical_price_data(self.start_date, self.end_date, "daily")
            df = pd.DataFrame(raw_data[i]['prices'])[['formatted_date','adjclose']]
            
            if k == 0:
                self.DF['Date' ] = df['formatted_date']

            df.columns = ['Date1',i]
            df['Date1']= pd.to_datetime(df['Date1'])

            self.DF['Date']= pd.to_datetime(self.DF['Date'])

            self.DF = self.DF.merge( df, how='left', left_on='Date',right_on='Date1')
            self.DF = self.DF.drop(labels='Date1', axis=1)

        #Renaming columns to represent instrument names rather than their ticker codes for ease of readability
        names.insert(0,'Date')
        self.DF.columns = names
        print('\n AFTER YAHOOFINANCE {} \n'.format( len( self.DF ) ))


    def fill_na( self ):

        #Front filling the NaN values in the data set
        self.DF = self.DF.fillna(method="ffill",axis=0)
        self.DF = self.DF.fillna(method="bfill",axis=0)

        self.cols = self.DF.columns.drop( 'Date' )
        self.DF[ self.cols ] = self.DF[ self.cols ].apply( pd.to_numeric, errors = 'coerce' ).round( decimals = 4 )



    def add_st_returns( self, change_days = [ 1, 3, 5, 14, 21 ] ):
        # Calculating Short term -Historical Returns for all features
        data = pd.DataFrame(data=self.DF['Date'])
        for i in change_days:
            x= self.DF[ self.cols ].pct_change( periods = i ).add_suffix( "-T-"+str( i ) )
            data=pd.concat(objs=(data,x),axis=1)
            x=[]
        self.train_data = data

        self.df_tech['Date'] = pd.to_datetime( self.df_tech['Date'] )
        self.train_data['Date'] = pd.to_datetime(self.train_data['Date'])
        self.train_data = pd.merge( left = self.train_data, right = self.df_tech, how = 'left', on = 'Date' )

    

    def add_volatility( self, windows = [ 5, 15, 30, 60, 90, 180 ] ):
        


        volatility = pd.DataFrame( self.DF[ 'Date' ],columns = [ 'Date' ] )
        volatility['Date']=pd.to_datetime(volatility['Date'],format='%Y-%b-%d')
        
        #Calculating volatilities for SPX
        for range in windows:
            volatility[ 'SPX/vol' + str( range ) ] = self.DF[ 'SPX' ].rolling( window = range ).std()-1

        self.train_data[ 'Date' ] = pd.to_datetime( self.train_data[ 'Date' ], format = '%Y-%b-%d' )
        self.train_data = pd.merge( left = self.train_data, right = volatility, how = 'left', on = 'Date' )


    def add_lt_returns( self, change_days = [ 60, 90, 180, 250 ] ):
        self.max_for_na = str( max( change_days ) )
        # Calculating Long term Historical Returns
        for i in change_days:
            x= self.DF[ self.features_for_returns ].pct_change( periods = i ).add_suffix( "-T-" + str( i ) )
            self.train_data = pd.concat( objs = ( self.train_data, x ), axis = 1 )
            x=[]


    def add_moving_average( self, sma_list = [ 15, 30, 60, 90, 180 ], ema_list = [ 90, 180 ] ):
        moving_avg = pd.DataFrame( self.DF[ 'Date' ], columns = [ 'Date' ] )

        moving_avg[ 'Date' ] = pd.to_datetime( self.DF[ 'Date' ], format = '%Y-%b-%d' )
        
        #Calculating volatilities for SPX
        for range in sma_list:
            moving_avg[ 'SPX/' + str( range ) + 'SMA' ] = ( self.DF[ 'SPX' ] / ( self.DF[ 'SPX' ].rolling( window = range ).mean() ) ) - 1

        for range in ema_list:
            moving_avg[ 'SPX/' + str( range ) + 'EMA' ] = ( self.DF[ 'SPX' ] / (self.DF[ 'SPX' ].ewm( span = 180,
                                                                                                      adjust = True,
                                                                                                      ignore_na = True ).mean() ) ) - 1
        moving_avg = moving_avg.dropna(axis=0)

        self.train_data[ 'Date' ] = pd.to_datetime( self.train_data[ 'Date' ], format = '%Y-%b-%d' )
        self.train_data = pd.merge( left = self.train_data, right = moving_avg, how = 'left', on = 'Date' )
        


    def add_regime_probs( self, low = False, high = True ):

        start_date = "2000-01-01"
        end_date = "2020-06-19"
        DF = pd.DataFrame()
        date_range = pd.bdate_range(start=start_date,end=end_date)
        values = pd.DataFrame({ 'Date': date_range})
        DF['Date']= pd.to_datetime(values['Date'])
        #Extracting Data from Yahoo Finance and Adding them to Values table using date as key
        raw_data = YahooFinancials('^GSPC')
        raw_data = raw_data.get_historical_price_data(start_date, end_date, "daily")
        df = pd.DataFrame(raw_data['^GSPC']['prices'])[['formatted_date','adjclose']]
        df.columns = ['Date1','SPY']
        df['Date1']= pd.to_datetime(df['Date1'])
        DF = DF.merge(df,how='left',left_on='Date',right_on='Date1')
        DF = DF.drop(labels='Date1', axis=1)
        DF['Returns'] = DF['SPY'].pct_change( periods = 1 )
        DF = DF[1:-1]
        rosemary = sm.tsa.MarkovRegression(DF['Returns'], k_regimes=2, trend='nc', switching_variance=True)
        rosemary_fitted = rosemary.fit()


        ddf = pd.DataFrame()
        ddf['Date'] = DF['Date']
        if low:
            ddf['regime_low'] = rosemary_fitted.smoothed_marginal_probabilities[0]
            ddf[ 'regime_low_5' ] = ddf['regime_low'].shift( 5 )
            ddf[ 'regime_low_10' ] = ddf['regime_low'].shift( 10 )
            ddf[ 'regime_low_20' ] = ddf['regime_low'].shift( 20 )

            self.the_list.append( 'regime_low' )
            self.the_list.append( 'regime_low_5' )
            self.the_list.append( 'regime_low_10' )
            self.the_list.append( 'regime_low_20' )

        if high:
            ddf['regime_high'] = rosemary_fitted.smoothed_marginal_probabilities[1]
            ddf[ 'regime_high_5' ] = ddf['regime_high'].shift( 5 )
            ddf[ 'regime_high_10' ] = ddf['regime_high'].shift( 10 )
            ddf[ 'regime_high_20' ] = ddf['regime_high'].shift( 20 )

            self.the_list.append( 'regime_high' )
            self.the_list.append( 'regime_high_5' )
            self.the_list.append( 'regime_high_10' )
            self.the_list.append( 'regime_high_20' )
        self.train_data = pd.merge( left = self.train_data, right = ddf, how = 'inner', on = 'Date', suffixes = ( False, False ) )


    def create_labels( self, target_return_period = 14, tail_probs = [ 0.25, 0.7 ] ):

        #Calculating forward returns for Target
        y = pd.DataFrame( data = self.DF[ 'Date' ] )

        y[ 'Target_Return' ] = self.DF[ 'SPX' ].pct_change( periods = -target_return_period ) * -1
        y['Close'] = self.DF[ 'SPX' ]
        y.isna().sum()

        # Removing NAs
        self.train_data = self.train_data[ self.train_data[ 'SPX-T-' + self.max_for_na ].notna() ]
        y = y[ y[ 'Target_Return' ].notna() ]


        #Adding Target Variables
        self.train_data = pd.merge( left = self.train_data, right = y, how = 'inner', on = 'Date', suffixes = ( False, False ) )
        self.train_data.isna().sum()

        #Select Threshold p (left tail probability)
        p1, p2 = tail_probs

        #Get z-Value
        z1 = st.norm.ppf( p1 )
        z2 = st.norm.ppf( p2 )
        

        #Calculating Threshold (t) for each Y
        crash = round( ( z1 * np.std( self.train_data[ "Target_Return" ] ) ) + np.mean( self.train_data[ "Target_Return" ] ), 5 )
        spike = round( ( z2 * np.std( self.train_data[ "Target_Return" ] ) ) + np.mean( self.train_data[ "Target_Return" ] ), 5 )

        print('crash threshold {} \n'.format( crash ) )
        print('spike threshold {} \n'.format( spike ) )

        #Creating Labels
        self.train_data[ 'crash' ] = ( self.train_data[ "Target_Return" ] < crash ) * 1
        self.train_data[ 'spike' ] = ( self.train_data[ "Target_Return" ] > spike ) * 1




        self.DATES = self.train_data[ 'Date' ]
        self.returns = list()

        self.returns = self.train_data[ 'Close' ].diff( periods = -14 ) * -1
        self.close = self.train_data[ 'Close' ]


        self.target_returns = self.train_data[ 'Target_Return' ]
        self.train_data = self.train_data.drop([ 'Target_Return', 'Date', 'Close' ], axis = 1 )
        self.train_data.drop( self.train_data.columns.difference( self.the_list ), 1, inplace=True)


    def add_garch( self ):
        df = self.DF[ ['Date', 'SPX'] ]
        df['pct_change'] = df[ 'SPX' ].pct_change()

        df[ 'stdev14' ] = df[ 'pct_change' ].rolling( window = 14, center = False ).std()
        df[ 'hvol14' ] = df[ 'stdev14' ] * ( 252 ** 0.5 )
        df = df.dropna()

        returns = df['pct_change'] * 100
        am = arch.arch_model(returns)

        res = am.fit(disp='off')

        df['forecast_vol'] = 0.1 * np.sqrt(res.params['omega'] + res.params['alpha[1]'] * res.resid**2 + res.conditional_volatility**2 * res.params['beta[1]'])


        df = df.drop(['pct_change', 'stdev14', 'hvol14'], axis = 1)

        df['forecast_vol'] = df['forecast_vol'].diff()
        self.train_data = pd.merge( left = self.train_data, right = df, how = 'inner', on = 'Date', suffixes = ( False, False ) )


    def add_kurt( self ):
        df = self.train_data[ [ 'Date', 'SPX-T-1' ] ]  
        df['kurt'] = df['SPX-T-1'].rolling(100).kurt()-3
        df['kurt_5'] = df['kurt'].shift( 2 )
        df['kurt_10'] = df['kurt'].shift( 3 )
        df['kurt_20'] = df['kurt'].shift( 5 )

        df = df.drop( 'SPX-T-1', axis = 1 )

        self.train_data = pd.merge( left = self.train_data, right = df, how = 'inner', on = 'Date', suffixes = ( False, False ) )
        self.the_list.append( 'kurt' )
        self.the_list.append( 'kurt_5' )
        self.the_list.append( 'kurt_10' )
        self.the_list.append( 'kurt_20' )



    def add_skew( self ):
        df = self.train_data[ [ 'Date', 'SPX-T-1' ] ]  
        df['skew'] = df['SPX-T-1'].rolling(100).skew()
        df['skew_5'] = df['skew'].shift( 5 )
        df['skew_10'] = df['skew'].shift( 10 )
        df['skew_20'] = df['skew'].shift( 20 )

        df = df.drop( 'SPX-T-1', axis = 1 )

        self.train_data = pd.merge( left = self.train_data, right = df, how = 'inner', on = 'Date', suffixes = ( False, False ) )
        self.the_list.append( 'skew' )
        self.the_list.append( 'skew_5' )
        self.the_list.append( 'skew_10' )
        self.the_list.append( 'skew_20' )


    def add_technical_ind( self ):
        raw_data = YahooFinancials( '^GSPC' )

        start_date = self.DF[ 'Date' ].iloc[ 0 ]
        end_date = self.DF[ 'Date' ].iloc[ -1 ]

        start_date = datetime.strftime(start_date, "%Y-%m-%d")
        end_date = datetime.strftime(end_date, "%Y-%m-%d")

        raw_data = raw_data.get_historical_price_data( start_date, end_date, "daily")

        df = pd.DataFrame(raw_data['^GSPC']['prices'])[ [ 'formatted_date','adjclose', 'open', 'high', 'low', 'close', 'volume' ]]

        self.df_tech = pd.DataFrame()
        self.df_tech['Date'] = df['formatted_date']

        self.df_tech['RSI'] = df.ta.rsi()
        self.df_tech['RSI_5'] = self.df_tech['RSI'].shift( 5 )
        self.df_tech['RSI_10'] = self.df_tech['RSI'].shift( 10 )
        self.df_tech['RSI_20'] = self.df_tech['RSI'].shift( 20 )

        self.df_tech.set_index( 'Date' )

        self.the_list.append( 'RSI' )
        self.the_list.append( 'RSI_5' )
        self.the_list.append( 'RSI_10' )
        self.the_list.append( 'RSI_20' )


    def add_trends( self ):
        
        df = dailydata.get_daily_data('recession', 2004, 1, 2019, 10, geo = 'USA')
        self.train_data = pd.merge( left = self.train_data, right = df, how = 'inner', on = 'Date', suffixes = ( False, False ) )
        self.the_list.append( 'kurt' )

    

class Classifierz():
    def __init__( self,  data_obj,  ttype = 'spike' ):
        super().__init__()
        self.class_list = list()
        self.class_key = list()
        self.metrics = [ f1_score, precision_score, roc_auc_score ]
        self.metircs_key = [ 'f1', 'precision', 'roc_auc']
        to_drop = list() 

        if ttype == 'spike':
            to_drop.append( 'crash' )
        else:
            to_drop.append( 'spike' )

        self.regimes = pd.DataFrame( data_obj.train_data[ to_drop[ : -1 ] ], columns = to_drop[ : -1 ], index = data_obj.train_data.index  )
        self.data = data_obj.train_data.drop( to_drop, axis = 1 ).copy()

        self.DATES = data_obj.DATES
        self.ALL_RETURNS = data_obj.returns

    def add_classifier( self, classifier, key ):
        self.class_list.append( classifier )
        self.class_key.append( key )

    def test(self):
        self.make_pred()
        self.scores()

    def train( self ):
        for classifier in self.class_list:
            classifier.fit( np.array(self.X_train), np.array(self.y_train) )


    def prepare_data( self, last_x_days, use_val_data = False, smotetomek = False, cs = 'spike'  ):

        self.DATE_TEST = self.DATES.iloc[ -last_x_days : ]
        self.TEST_RETURNS = self.ALL_RETURNS[ -last_x_days : ].to_list()



        X = self.data.drop( [ cs ], axis = 1 )
        y = self.data[ cs ]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split( X, y, test_size = last_x_days / len( y ), random_state = 103, stratify = y ) 
        X__ = self.X_train
        if use_val_data:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split( self.X_train, self.y_train, test_size = 0.2, random_state = 103, stratify = self.y_train ) 

        if smotetomek:
            self.smotetomek()

        mm = MinMaxScaler()
        mm = mm.fit( self.X_train )
        self.X_train = mm.transform(self.X_train)

        if use_val_data:
            self.X_val = mm.transform(self.X_val)

        mm_NEW = MinMaxScaler()
        mm_NEW = mm_NEW.fit( X__ )
        self.X_test = mm_NEW.transform( self.X_test )

    def scores( self ):
        KEYS = self.results.keys().to_list()
        counter = 0
        for  key in KEYS:
            if key != 'Target':
                score_list = list()
                for metric in self.metrics:
                    score_list.append( metric( self.results[ 'Target' ], self.results[ key ] ) )

                if counter == 0:
                    scores = pd.DataFrame( { key: score_list  })
                    counter = counter + 1
                else:
                    scores[ key ] = score_list
                    counter = counter + 1
        self.scores = scores
    

    def make_pred( self, for_valid = True ):
        if for_valid: 
            TARGET = self.y_val
            XXX = self.X_val
        else:
            TARGET = self.y_test
            XXX = self.X_test

        for i, classifier in enumerate(  self.class_list ):
            temp = classifier.predict( XXX )
            if i == 0:
                results = pd.DataFrame( { 'Target': TARGET,
                                            self.class_key[ i ] : temp } )
            else:
                results[ self.class_key[ i ] ] = temp
       
        if for_valid: 
            self.results = results
        else:
            if self.counter == 0:
                self.BT_RESULTS = results
            else: 
                self.BT_RESULTS.append( results )
            self.counter = self.counter + 1

    def backtest(self):
        list_of_cash_tt = list()
        self.make_pred( for_valid = False )
        classifiers = self.BT_RESULTS.keys().to_list()
        count = 0
        for CCC in classifiers:
            if CCC != 'Target':
                temp_cash = list()
                temp_cash.append( 0 )
                cc = 0

                preds = self.BT_RESULTS[ CCC ].to_list()

                assert len( preds ) ==  len( self.TEST_RETURNS )

                for i in range( len( preds ) ):
                    if preds[ i ] == 1:
                        cc += self.TEST_RETURNS[ i ]
                    temp_cash.append( cc )

                if count == 0:
                    CASH = pd.DataFrame( { CCC: temp_cash  })
                    count += 1
                else:
                    CASH[ CCC ] = temp_cash
                    count += 1

        self.CASH = CASH

    def smotetomek( self ):

        smt = SMOTETomek()
        self.X_train, self.y_train = smt.fit_sample( self.X_train, self.y_train )

