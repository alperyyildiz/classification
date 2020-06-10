probs_of_c1 = results['Stack'].to_list()
probs_of_c1 = np.round(np.array(probs_of_c1))
date_list = TEST_df['Date'].dt.date.to_list()

BT = pd.DataFrame( { 'Date': date_list,
                     'pred': probs_of_c1,
                     'Target' : results['Target'].to_list(),
                     'T/F' : probs_of_c1 == results['Target'].to_list()
                    } )
#Initialize cash amount and init state 
cash = 1000
cash_real = 1000

state = 'CASH'
state_real = 'CASH'
#Get # of test days and binary predictions 
num_of_samples = BT.shape[ 0 ]
print('NUM OF SAMPLES --> {}'.format( num_of_samples ) )
PREDS = BT[ 'pred' ].to_list()
TARGET = BT[ 'Target' ].to_list()


#Get close prices 

#Init signals as np.array all 0 
SIGNAL = np.zeros(num_of_samples)
SIGNAL_REAL = np.zeros(num_of_samples)

#Get first buy signal
first_buy_signal_index = find_first(  np.array( PREDS ) == 1 )


for ind in range( first_buy_signal_index, num_of_samples ):
    if state == 'CASH' and PREDS[ ind - 4 ] == 1:
        SIGNAL[ ind ] = 1
        state = 'IN'
    elif state == 'IN' and PREDS[ ind - 4] == 0:
        SIGNAL[ ind ] = -1
        state = 'CASH'

    if state_real == 'CASH' and  TARGET[ ind - 4 ] == 1:
        SIGNAL_REAL[ ind ] = 1
        state_real = 'IN'
    elif state_real == 'IN' and TARGET[ ind - 4 ] == 0:
        SIGNAL_REAL[ ind ] = -1
        state_real = 'CASH'


trading_returns = np.zeros( num_of_samples )
trading_returns_real = np.zeros( num_of_samples )



for ind in range( num_of_samples ):
    if SIGNAL[ ind ] == 1:
        price_buy = Close[ ind ]
    elif SIGNAL[ ind ] == -1:
        trading_returns[ ind ] = Close[ ind ] / price_buy - 1  

for ind in range( num_of_samples ):
    if SIGNAL_REAL[ ind ] == 1:
        price_buy = Close[ ind ]
    elif SIGNAL_REAL[ ind ] == -1:
        trading_returns_real[ ind ] = Close[ ind ] / price_buy - 1  

for ind in range( num_of_samples ):
    if trading_returns[ ind ] != 0:
        cash = cash + cash * trading_returns[ ind ]

for ind in range( num_of_samples ):
    if trading_returns_real[ ind ] != 0:
        cash_real = cash_real + cash_real * trading_returns_real[ ind ]

BT[ 'Buy Sell Signals'] = Signals
BT[ 'Close' ] = Close
BT[ 'Returns'] = trading_returns
BT[ 'Returns with True Labels'] = trading_returns_real
