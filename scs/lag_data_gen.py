class LaggedTimeSeriesDF():
    def __init__(self, df, lag=None):
        self.df = df

        if lag is not None:
            self.t = lag
            self.df = self.__apply_lags__()
    
    def __apply_lags__(self):
        new_df = self.df.copy(deep=True).dropna()
        col_names = self.df.columns.values.tolist()
        # generate lagged array with t lags
        for col_name in col_names:
                new_df[col_name + '_lag' + str(self.t)] = self.df[col_name].shift(self.t)

        return new_df.iloc[self.t:]