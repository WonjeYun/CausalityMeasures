import numpy as np
import ols_gc



class GrangerCausality():
    def __init__(self, lts_array, endog, exog, lag = None):
        ''' input LaggedTimeSeriesDF object as lts_df'''
        self.array = lts_array
        self.endog = endog  # Dependent Variable Y
        self.exog = exog  # Independent Variable X
        self.lag = lag

    def Granger_Caus(self, df=None, n_shuffles=0):
        ## Prepare lists for storing results
        granger_causalities = [0,0]
        GCs = []
        shuffled_TEs = []
        p_values = []
        z_scores = []

        df = np.copy(self.array)

        ## Require us to compare information transfer bidirectionally
        for i in range(2):
            ## Calculate Residuals after OLS Fitting, for both Independent and Joint Cases
            joint_residuals = ols_gc.ols_res_cal(df[:, i], df[:, [i+2, 3-i]])
            independent_residuals = ols_gc.ols_res_cal(df[:, i], df[:, i+2].reshape(-1, 1))

            ## Use Geweke's formula for Granger Causality 
            granger_causalities[i] = ols_gc.granger_cal(independent_residuals, joint_residuals)

        GCs.append(granger_causalities)
        ## Calculate Significance of GC during this window
        if n_shuffles > 0:
            p, z, TE_mean = significance(df=df,
                                            TE=granger_causalities,
                                            endog=self.endog,
                                            exog=self.exog,
                                            lag=self.lag,
                                            n_shuffles=n_shuffles,
                                            method='granger_causality')

            shuffled_TEs.append(TE_mean)
            p_values.append(p)
            z_scores.append(z)
            # column are [XY, YX]
            # rows are [TE, p_value, z_score, shuffled_TE]
            self.results = np.concatenate(
                (np.array(GCs), np.array(p_values), np.array(z_scores), np.array(shuffled_TEs)), axis=0)
        else:
            ## Store Granger Causality from X(t)->Y(t) and from Y(t)->X(t)
            self.results = np.array(GCs)

        return self.results
    

def significance(df, TE, endog, exog, lag, n_shuffles, method, bandwidth=None):
    """
        Perform significance analysis on the hypothesis test of statistical causality, for both X(t)->Y(t)
        and Y(t)->X(t) directions
   
        Calculated using:  Assuming stationarity, we shuffle the time series to provide the null hypothesis. 
                           The proportion of tests where TE > TE_shuffled gives the p-value significance level.
                           The amount by which the calculated TE is greater than the average shuffled TE, divided
                           by the standard deviation of the results, is the z-score significance level.

        Arguments:
            TE              -      (list)    Contains the transfer entropy in each direction, i.e. [TE_XY, TE_YX]
            endog           -      (string)  The endogenous variable in the TE analysis being significance tested (i.e. X or Y) 
            exog            -      (string)  The exogenous variable in the TE analysis being significance tested (i.e. X or Y) 
            pdf_estimator   -      (string)  The pdf_estimator used in the original TE analysis
            bins            -      (Dict of lists)  The bins used in the original TE analysis

            n_shuffles      -      (float) Number of times to shuffle the dataframe, destroyig temporality
            both            -      (Bool) Whether to shuffle both endog and exog variables (z-score) or just exog                                  variables (giving z*-score)  
        Returns:
            p_value         -      Probablity of observing the result given the null hypothesis
            z_score         -      Number of Standard Deviations result is from mean (normalised)
        """

    ## Prepare array for Transfer Entropy of each Shuffle
    shuffled_TEs = np.zeros(shape=(2, n_shuffles))

    for i in range(n_shuffles):
        ## Perform Shuffle
        df = shuffle_along_axis(df, axis=0)

        if method == 'granger_causality':
            ## Calculate New TE
            shuffled_causality = GrangerCausality(df, endog=endog, exog=exog, lag=lag)
            TE_shuffled = shuffled_causality.Granger_Caus(df, n_shuffles=0)

    ## Calculate p-values for each direction
    p_values = (np.count_nonzero(TE[0] < shuffled_TEs[0, :]) / n_shuffles, \
                np.count_nonzero(TE[1] < shuffled_TEs[1, :]) / n_shuffles)

    shuff_te_zero = np.std(shuffled_TEs[0, :]) + np.finfo(float).eps
    shuff_te_one = np.std(shuffled_TEs[1, :]) + np.finfo(float).eps

    ## Calculate z-scores for each direction
    z_scores = ((TE[0] - np.mean(shuffled_TEs[0, :])) / shuff_te_zero, \
                (TE[1] - np.mean(shuffled_TEs[1, :])) / shuff_te_one)

    TE_mean = (np.mean(shuffled_TEs[0, :]), \
               np.mean(shuffled_TEs[1, :]))

    ## Return the self.DF value to the unshuffled case
    return p_values, z_scores, TE_mean

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)