#THIS IS TRADING ENVIRONMENT
#THIS IS TRADING ENVIRONMENT
#THIS IS TRADING ENVIRONMENT
#THIS IS TRADING ENVIRONMENT


class Trade_Env(gym.Env):
    """Class for a discrete (buy/hold/sell) spread trading environment.
    """

    def __init__(self, data_generator, episode_length=1000,
                 trading_fee=0, time_fee=0, no_pos_fee=0, profit_taken=20, stop_loss=-10,
                 reward_factor=10000, lookback = 2):
        """Initialisation function
        Args:
            data_generator (tgym.core.DataGenerator): A data generator object yielding a 1D array of bid-ask prices.
            episode_length (int): number of steps to play the game for
            trading_fee (float): penalty for trading
            time_fee (float): time fee
            lookback (int): number of historical states to stack in the observation vector.
       
        positons:
                  0 for flat
                  1 for long
                  2 for short

        actions:  
                  0 for hold
                  1 for buy
                  2 for sell
        """

        assert lookback > 0
        self._data_generator = data_generator
        #self._first_render = True
        self._trading_fee = trading_fee
        self._time_fee = time_fee
        self._no_pos_fee = no_pos_fee
        self._episode_length = episode_length - lookback
        self.n_actions = 3
        self._lookback = lookback
        self._tick_buy = 0
        self._tick_sell = 0
        self._max_lost = -1000


        self.reset()
        self.action_space = spaces.Discrete( self.n_actions )
        self.observation_space = spaces.Box( low=-100000, high=200000, shape=( self._lookback, 4 ), dtype=np.float32)
        
        #self.tick_cci_14 = 0
        #self.tick_rsi_14=0
        #self.tick_dx_14 = 0
        #self.TP_render=False
        #self.SL_render = False
        #self.Buy_render=False
        #self.Sell_render=False
        #self.tick_mid = 0 # saeed

    def reset(self):
        """Reset the trading environment. Reset rewards, data generator...
        Returns:
            observation (numpy.array): observation of the state
        """

        self._iteration = 0

        #data generator object to its initial position
        #meaning it resets the iterator index to 0
        self._data_generator.rewind()

        #reset reward/pnl/position
        #self._total_reward = 0
        #self._total_pnl = 0
        # self._entry_price = 0
        # self._exit_price = 0

        #HOLD - HOLD - HOLD
        self._position = 0

        self._closed_plot = False
        self._max_lost = -1000

        #regenerate history data
        for i in range(self._lookback):
            generated = (next(self._data_generator).reshape(1,-1) )
            if i == 0:
                self._historical_data = generated.copy()
            elif i == self._lookback-1:
                self.current_price = generated[ 0 ]
                self._historical_data = np.concatenate( (self._historical_data, generated ), axis = 0 )
            else:
                self._historical_data = np.concatenate( (self._historical_data, generated ), axis = 0 )


        #--------------------FOR RENDER-------------------------#
        #self._tick_buy, self._tick_sell,self.tick_mid ,self.tick_rsi_14,self.tick_cci_14 = self._historical_data[0][:5]

        #gets self._historical_data as array
        observation = self._get_observation()

        print( observation.shape )

        #defines initial action as hold
        self._action = 0

        return observation



    def step(self, action):
        """Take an action (buy/sell/hold) and computes the immediate reward.
        Args:
            action (numpy.array): Action to be taken, one-hot encoded.
        Returns:
            tuple:
                - observation (numpy.array): Agent's observation of the current environment.
                - reward (float) : Amount of reward returned after previous action.
                - done (bool): Whether the episode has ended, in which case further step() calls will return undefined results.
                - info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """


        self._action = action
        self._iteration += 1
        done = False
        info = {}

        #PAY TIME FEE FOR DOING NOTHING -------------------------- motivate algo to do something


        reward = self._get_reward( action )

        #THESE FOR RENDER FOR RENDER FOR RENDER 
        #self._total_pnl += instant_pnl
        #self._total_reward += reward
        print( 'REWARD \n {}'.format(reward) )
        try:
            raw = next(self._data_generator)
            self.current_price = raw[ 0 ]
            self._historical_data = np.concatenate( (self._historical_data, (raw.reshape(1,-1) ) ), axis = 0 )

            #--------------------FOR RENDER-------------------------#
            #self._tick_sell, self._tick_buy, self.tick_mid, self.tick_rsi_14, self.tick_cci_14 = self._historical_data[-1][:5]
        
        except StopIteration:
            done = True
            info['status'] = 'No more data.'

        # Game over logic
        if self._iteration >= self._episode_length:
            done = True
            info['status'] = 'Time out.'
        if reward <= self._max_lost:
            done = True
            info['status'] = 'Bankrupted.'
        if self._closed_plot:
            info['status'] = 'Closed plot'

        observation = self._get_observation()

        return observation, reward, done, info

    def _init_pos_act_var( self, action ):
        HOLD = False
        BUY = False
        SELL = False

        FLAT = False
        LONG = False
        SHORT = False

        if action == 0:
            HOLD = True
        elif action == 1:
            BUY = True
        else:
            SELL = True

        if self._position == 0:
            FLAT = True
        elif self._position == 1:
            LONG = True
        else:
            SHORT = True
        return HOLD, BUY, SELL, FLAT, LONG, SHORT
        


    def _get_reward( self, action ):

        HOLD, BUY, SELL, FLAT, LONG, SHORT = self._init_pos_act_var( action )

        
        ACT_PNL = 0
        INSTANT_PNL = 0
        REWARD = -self._time_fee

        if BUY or SELL:
            #PAY TRADING FEE 
            REWARD -= self._trading_fee


        #ACTION BUY
        if BUY:

            #IF POSITION IS FLAT (WHEN WE BUY)
            if FLAT:
                
                #SAVE POSITION AS LONG
                self._position = 1
                self._entry_price = self.current_price
                #self.Buy_render = True

            #IF POSITION IS SHORT (WHEN WE BUY)
            elif SHORT:
                #SAVE POSITION AS FLAT
                self._position = 0

                #CALCULATE TRADING REWARD
                ACT_PNL = self._entry_price - self.current_price
                self._entry_price = 0
                # self.Buy_render = True
                #if (ACT_PNL > 0):
                #    self.TP_render=True
                #else:
                #    self.SL_render=True

        #ACTION SELL
        elif SELL:

            #IF POSITION IS FLAT (WHEN WE SELL)
            if FLAT:
                self._position = 2
                self._entry_price = self.current_price
                #self.Sell_render = True

            #IF POSITION IS LONG (WHEN WE SELL)
            elif LONG:
                ACT_PNL = self.current_price - self._entry_price
                self._position = 0
                self._entry_price = 0
                # self.Sell_render = True
                #if (ACT_PNL > 0):
                #    self.TP_render = True
                #else:
                #    self.SL_render = True

        #ACTION HOLD
        elif HOLD:

            if FLAT:
                REWARD -= self._no_pos_fee

            elif SHORT:
                INSTANT_PNL = self._entry_price - self.current_price
            elif LONG:
                INSTANT_PNL = self.current_price - self._entry_price

        #else:
            #self.Buy_render = self.Sell_render = False
            #self.TP_render = self.SL_render = False

        print( '\n POS {} and ACT {} \n  \n  '.format( self._position, action ) )
        REWARD = REWARD + ACT_PNL + INSTANT_PNL

        print( '\n REWARD {} \n  \n  '.format( REWARD ) )

        return REWARD

    def _handle_close(self, evt):
        self._closed_plot = True

  

    def _get_observation(self):
        """Concatenate all necessary elements to create the observation.
        Returns:
            numpy.array: observation array.
        """


        return self._historical_data[ -self._lookback:, 1: ]
          
    @staticmethod
    def random_action_fun():
        """The default random action for exploration.
        We hold 80% of the time and buy or sell 10% of the time each.
        Returns:
            numpy.array: array with a 1 on the action index, 0 elsewhere.
        """
        return np.random.multinomial(1, [0.8, 0.1, 0.1])
