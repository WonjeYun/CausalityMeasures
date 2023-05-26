import numpy as np

class LaggedTimeSeriesArray():
    def __init__(self, df, lag=None):
        self.df = df

        if lag is not None:
            self.t = lag
            self.df = self.__apply_lags__()
    
    def __apply_lags__(self):
        new_df = np.copy(self.df)
        new_df = new_df[~np.isnan(new_df).any(axis=1)]
        # generate lagged array with t lags
        new_df = np.append(new_df, np.roll(new_df, self.t, axis=0), axis=1)

        return new_df[2:]