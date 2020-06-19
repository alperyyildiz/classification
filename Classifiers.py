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
from pycaret.classification import *
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTETomek
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, roc_auc_score

ticker_details = pd.read_excel('Ticker List.xlsx')
class Classifierz():
    def __init__( self ):
        self.class_list = list()
        self.class_key = list()
        self.metrics = [ f1_score, precision_score, roc_auc_score ]
        self.metircs_key = [ 'f1', 'precision', 'roc_auc']
        #self.first = True 


    def add_classifier( self, classifier, key ):
        self.class_list.append( classifier )
        self.class_key.append( key )

    def test(self):
        self.make_pred()
        self.scores()


    def make_pred( self, for_valid = True ):
        if for_valid: 
            TARGET = self.y_val
            XXX = self.X_val
        else:
            TARGET = self.y_test
            XXX = self.X_test

        for i, classifier in enumerate(  self.class_list ):
            print(i)
            temp = classifier.predict( XXX )
            if i == 0:
                results = pd.DataFrame( { 'Target': TARGET,
                                            self.class_key[ i ] : temp } )
            else:
                results[ key ] = temp
    
        self.results = results


    def split( self, last_x_days ):
        self.TEST_data = self.data.iloc[ -last_x_days : ]
        self.TRAIN_data = self.data.iloc[ : -last_x_days ]

        X = self.TRAIN_data.drop(['Y-141'],axis=1)
        y = self.TRAIN_data['Y-141']
        
        

        X_train, self.X_val, y_train, self.y_val = train_test_split( X, y, test_size = 0.3 ) 
        smt = SMOTETomek()

        self.X_train, self.y_train = smt.fit_sample( X_train, y_train )

        self.X_test = self.TEST_data.drop(['Y-141'],axis=1)
        self.y_test = self.TEST_data['Y-141']

    def train( self ):
        for classifier in self.class_list:
            classifier.fit( self.X_train, self.y_train )
            print('JHEY')

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
                    scores[ key ] = score
                    counter = counter + 1
        self.scores = scores

    def preprocess( self, ticker_details ):
        ticker = ticker_details['Ticker'].to_list()
        names = ticker_details['Description'].to_list()

        #Extracting Data from Yahoo Finance and Adding them to Values table using date as key
        end_date= "2020-03-01"
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
        #print(data.shape)



        # Calculating Long term Historical Returns
        change_days = [60,90,180,250]

        for i in change_days:
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
        print(moving_avg.shape)
        #print(moving_avg.head())

        #Merging Moving Average values to the feature space
        #print(data.shape)
        data['Date']=pd.to_datetime(data['Date'],format='%Y-%b-%d')
        data = pd.merge(left=data,right=moving_avg,how='left',on='Date')
        #print(data.shape)
        data.isna().sum()

        #Caluculating forward returns for Target
        y = pd.DataFrame(data=values['Date'])

        y['SPX-T+14']=values["SPX"].pct_change(periods=-14)
        y['SPX-T+22']=values["SPX"].pct_change(periods=-22)
        y.isna().sum()

        # Removing NAs
        data = data[data['SPX-T-250'].notna()]
        y = y[y['SPX-T+22'].notna()]
        



        #Adding Target Variables
        data = pd.merge(left=data,right=y,how='inner',on='Date',suffixes=(False,False))
        data.isna().sum()

        #Select Threshold p (left tail probability)
        p1 = 0.25
        p2 = 0.75
        #Get z-Value
        z1 = st.norm.ppf(p1)
        z2 = st.norm.ppf(p2)
       
        #Calculating Threshold (t) for each Y
        t_141 = round((z1*np.std(data["SPX-T+14"]))+np.mean(data["SPX-T+14"]),5)
        t_142 = round((z2*np.std(data["SPX-T+14"]))+np.mean(data["SPX-T+14"]),5)
        t_221 = round((z1*np.std(data["SPX-T+22"]))+np.mean(data["SPX-T+22"]),5)
        t_222 = round((z2*np.std(data["SPX-T+22"]))+np.mean(data["SPX-T+22"]),5)

        #Creating Labels
        data['Y-141'] = (data['SPX-T+14']< t_141)*1
        data['Y-142'] = (data['SPX-T+14']> t_142)*1
        data['Y-221']= (data['SPX-T+22']< t_221)*1
        data['Y-222']= (data['SPX-T+22']> t_222)*1

        data = data.drop(['SPX-T+14','SPX-T+22','Date'],axis=1)

        self.data = data.drop(['Y-221','Y-222','Y-142'],axis=1)
