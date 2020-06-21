import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from yahoofinancials import YahooFinancials
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import seaborn as sns
import scipy.stats as st
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTETomek
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, roc_auc_score
import pandas_ta as ta

class HELP():
    def __init__( self ):
        pass
    def preprocess( self ):
        ticker_details = pd.read_excel('Ticker List.xlsx')
        ticker = ticker_details['Ticker'].to_list()
        names = ticker_details['Description'].to_list()

        #Extracting Data from Yahoo Finance and Adding them to Values table using date as key
        end_date= "2020-05-10"
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
        values.tail()


        #Front filling the NaN values in the data set
        values = values.fillna(method="ffill",axis=0)
        values = values.fillna(method="bfill",axis=0)
        values.isna().sum()

        #Return
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
        # Co-ercing numeric type to all columns except Date
        cols=values.columns.drop('Date')
        values[cols] = values[cols].apply(pd.to_numeric,errors='coerce').round(decimals=4)
        #print(values.tail())

        imp = ['Gold','USD Index', 'Oil', 'SPX','VIX', 'High Yield Fund' , 'Nikkei', 'Dax', '10Yr', '2Yr' , 'EEM' ,'XLE', 'XLF', 'XLI', 'AUDJPY']
        # Calculating Short term -Historical Returns
        change_days = [1,3,5,14,21]
        
        data = pd.DataFrame(data=values['Date'])
        for i in change_days:
            x= values[cols].pct_change(periods=i).add_suffix("-T-"+str(i))
            data=pd.concat(objs=(data,x),axis=1)
            x=[]



        # Calculating Long term Historical Returns
        change_days = [60,90,180,250]

        for i in change_days:
            x= values[imp].pct_change(periods=i).add_suffix("-T-"+str(i))
            data=pd.concat(objs=(data,x),axis=1)
            x=[]

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
        #Merging Moving Average values to the feature space
        data['Date']=pd.to_datetime(data['Date'],format='%Y-%b-%d')


        self.RAW_data = pd.merge(left=data,right=moving_avg,how='left',on='Date')

        self.RAW_y = pd.DataFrame(data=values['Date'])

        self.RAW_values = values
    
    def define_classes( self, return_days ):

        y = self.RAW_y
        values = self.RAW_values
        data = self.RAW_data

        y[ 'SPX-Target' ] = values[ "SPX" ].pct_change( periods = -return_days )
        y.isna().sum()

        # Removing NAs
        data = data[ data[ 'SPX-T-250' ].notna() ]
        y = y[ y[ 'SPX-Target' ].notna() ]

        #Adding Target Variables
        data = pd.merge(left=data,right=y,how='inner',on='Date',suffixes=(False,False))
        data.isna().sum()
        
        #Select Threshold p (left tail probability)
        p1 = 0.25
        p2 = 0.75
        #Get z-Value
        z1 = st.norm.ppf( p1 )
        z2 = st.norm.ppf( p2 )
       
        #Calculating Threshold (t) for each Y
        crash = round( (z1 * np.std( data[ "SPX-Target" ] ) ) + np.mean( data[ "SPX-Target" ] ), 5 )
        spike = round( (z2 * np.std( data[ "SPX-Target" ] ) ) + np.mean( data[ "SPX-Target" ] ), 5 )
        
        #Creating Labels
        data[ 'Y-Crash' ] = ( data[ 'SPX-Target' ] < crash ) * 1
        data[ 'Y-Spike' ] = ( data[ 'SPX-Target' ] > spike ) * 1

        #Save Dates
        self.DATE_ALL_DATA = data[ 'Date' ]
        last_values = values[ values[ 'Date' ] <= data[ 'Date' ].iloc[ -1 ] ]
        self.ALL_RETURNS = list()

        #Calculate cash returns
        RATES = data[ 'SPX-Target' ].to_list()
        CLOSE = last_values[ 'SPX' ].to_list()

        for i in range( len( data[ 'SPX-Target' ] ) ):
            self.ALL_RETURNS.append( RATES[ i ] * CLOSE[ i ] )

        #For now, data has Spike and Crash labels 
        self.data = data.drop( [ 'SPX-Target', 'Date' ], axis = 1 )

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
                        cc += -self.TEST_RETURNS[ i ]
                    temp_cash.append( cc )

                if count == 0:
                    CASH = pd.DataFrame( { CCC: temp_cash  })
                    count += 1
                else:
                    CASH[ CCC ] = temp_cash
                    count += 1

        self.CASH = CASH
      


    def scores( self, typez ):
        if typez == 'crash':
            results = self.results_crash
        else:
            results = self.results_spike

        KEYS = results.keys().to_list()
        counter = 0
        for  key in KEYS:
            if key != 'Target':
                score_list = list()
                for metric in self.metrics:
                    score_list.append( metric( results[ 'Target' ], results[ key ] ) )

                if counter == 0:
                    scores = pd.DataFrame( { key: score_list  })
                    counter = counter + 1
                else:
                    scores[ key ] = score_list
                    counter = counter + 1

        if typez == 'crash':
            self.scores_crash = scores
        else:
            self.scores_spike = scores



    def update_data( self ):

        #Update train-test split after every cycle time
        #if it is last cycle, use remaining days as test data
        if self.counter_max == self.counter:
            print('\n RANGE')
            print(-self.last_x_days + self.counter * self.cycle)
            print('\n')
            print(-self.last_x_days + self.counter * self.cycle + self.last_cycle)
            TEST_data = self.data.iloc[ -self.last_x_days + self.counter * self.cycle : ]
            TRAIN_data = self.data.iloc[ : -self.last_x_days + self.counter * self.cycle ]
        else:
            TEST_data = self.data.iloc[ -self.last_x_days + self.counter * self.cycle : -self.last_x_days + self.counter * self.cycle + self.cycle ]
            TRAIN_data = self.data.iloc[ : -self.last_x_days + self.counter * self.cycle ]


        #Create Training set and crash-spike labels
        X_train = TRAIN_data.drop( ['Y-Crash', 'Y-Spike'], axis = 1 )
        y_train_crash = TRAIN_data[ 'Y-Crash' ]
        y_train_spike = TRAIN_data[ 'Y-Spike' ]

        #Create test sets
        self.X_test = TEST_data.drop( [ 'Y-Spike', 'Y-Crash' ], axis = 1 )
        self.y_test_crash = TEST_data[ 'Y-Crash' ]
        self.y_test_spike = TEST_data[ 'Y-Spike' ]

        #Scale data between 0-1
        mm = MinMaxScaler()
        mm = mm.fit( X_train )
        X_train = mm.transform( X_train )
        self.X_test = mm.transform( self.X_test )

        #Create samples with smotetomek to eliminate imbalance of positive instances
        self.X_train_crash, self.y_train_crash = self.smotetomek( X_train, y_train_crash )
        self.X_train_spike, self.y_train_spike = self.smotetomek( X_train, y_train_spike )



class Classifierz( HELP ):
    def __init__( self, last_x_days, cycle, return_days ):
        super().__init__()
        self.class_list_spike = list()
        self.class_list_crash = list()
        self.class_key_spike = list()
        self.class_key_crash = list()

        self.metrics = [ f1_score, precision_score, roc_auc_score ]
        self.metircs_key = [ 'f1', 'precision', 'roc_auc']
        self.last_x_days = last_x_days
        self.cycle = cycle
        self.counter = 0
        self.counter_max = last_x_days // cycle -1 if last_x_days % cycle == 0 else last_x_days // cycle 
        self.last_cycle = self.cycle if last_x_days % cycle == 0 else last_x_days % cycle
        self.preprocess()
        self.define_classes( return_days = return_days )

        self.DATE_TEST = self.DATE_ALL_DATA.iloc[ -last_x_days : ]
        self.TEST_RETURNS = self.ALL_RETURNS[ -last_x_days : ]


    def make_pred( self, typez ):
        XXX = self.X_test
        if typez == 'crash':
            TARGET = self.y_test_crash
            classifiers = self.class_list_crash
            class_keys = self.class_key_crash
        else:
            TARGET = self.y_test_spike
            classifiers = self.class_list_spike
            class_keys = self.class_key_spike

        for i, classifier in enumerate( classifiers ):
            temp = classifier.predict( XXX )
            if i == 0:
                results = pd.DataFrame( { 'Target': TARGET, class_keys[ i ] : temp } )
            else:
                results[ class_keys[ i ] ] = temp
       

        if typez == 'crash':
            if self.counter == 0:
                self.results_crash = results
            else:
                self.results_crash.append( results )
        else:
            if self.counter == 0:
                self.results_spike = results
            else:
                self.results_crash.append( results )



    def add_classifier( self, classifier, key, typez ):
        if typez == 'crash':
            self.class_list_crash.append( classifier )
            self.class_key_crash.append( key )
        else:
            self.class_list_spike.append( classifier )
            self.class_key_spike.append( key )


    def smotetomek( self, X_train, y_train ):
        smt = SMOTETomek()
        return smt.fit_sample( X_train, y_train )

    def retrain( self ):

        for i in range( self.counter_max + 1 ):
            print( '\n TRAIN # {} \n'.format( i ) )
            self.update_data()
            for classifier in self.class_list_spike:
                classifier.fit( self.X_train_spike, self.y_train_spike )

            for classifier in self.class_list_crash:
                classifier.fit( self.X_train_crash, self.y_train_crash )
            print('\n FINISHED {} \n '.format( i ) )
            self.make_pred( typez = 'crash' )
            self.make_pred( typez = 'spike' )
            self.counter += 1


rosemary = Classifierz( last_x_days = 20, cycle = 10, return_days = 14 )


ext = ExtraTreesClassifier( n_estimators = 100, random_state = 0 )
gb = GradientBoostingClassifier( random_state = 0 )

rosemary.add_classifier( ext, 'EXT', 'crash' )
rosemary.add_classifier( gb, 'GB', 'crash')

rosemary.add_classifier( ext, 'EXT', 'spike' )
rosemary.add_classifier( gb, 'GB', 'spike')

rosemary.retrain()
