import numpy as np
from numpy import ma
from scipy import stats, linalg
from six import string_types


class TransferEntropyArray():
    def __init__(self, lts_array, endog, exog, lag = None, method=None):
        ''' input LaggedTimeSeriesDF object to numpy array lts_array'''
        self.array = lts_array
        self.endog = endog  # Dependent Variable Y
        self.exog = exog  # Independent Variable X
        self.lag = lag

        self.covars = self.covariance_cal(method)

    # @njit
    def covariance_cal(self, method):
        self.covars = [[], []]
        for i in range(2):

            if method == 'mi':
                self.covars[i] = [np.ones(shape=(1, 1)) * np.var(self.array[:, [1 - i]]),
                                  np.ones(shape=(1, 1)) * np.var(self.array[:, [i]]),
                                  np.cov(self.array[:, [1 - i, i]].T)]
            else:
                self.covars[i] = [np.cov(self.array[:, [i, i+2, 3-i]].T),
                                  np.cov(self.array[:, [3-i, i+2]].T),
                                  np.cov(self.array[:, [i, i+2]].T),
                                  np.ones(shape=(1, 1)) * np.var(self.array[:, [i+2]])]
        return self.covars
    
    def TransferEntropy(self, df=None, bandwidth=None, gridpoints=20, n_shuffles=0):
        ## Prepare lists for storing results
        TEs = []
        shuffled_TEs = []
        p_values = []
        z_scores = []

        df = np.copy(self.array)

        ## Initialise list to return TEs
        transfer_entropies = [0, 0]

        ## Require us to compare information transfer bidirectionally
        for i in range(2):
            ### Entropy calculated using Probability Density Estimation:
            ### Estimate PDF using Gaussian Kernels and use H(x) = p(x) log p(x)
            ## 1. H(Y,Y-t,X-t)
            H1 = self.get_entropy(df=df[:, [i, i+2, 3-i]],
                                gridpoints=gridpoints,
                                bandwidth=bandwidth,
                                covar=self.covars[i][0])
            ## 2. H(Y-t,X-t)
            H2 = self.get_entropy(df=df[:, [3-i, i+2]],
                                gridpoints=gridpoints,
                                bandwidth=bandwidth,
                                covar=self.covars[i][1])
            ## 3. H(Y,Y-t)
            H3 = self.get_entropy(df=df[:, [i, i+2]],
                                gridpoints=gridpoints,
                                bandwidth=bandwidth,
                                covar=self.covars[i][2])
            ## 4. H(Y-t)
            H4 = self.get_entropy(df=df[:, [i+2]],
                                gridpoints=gridpoints,
                                bandwidth=bandwidth,
                                covar=self.covars[i][3])

            ### Calculate Conditonal Entropy using: H(Y|X-t,Y-t) = H(Y,X-t,Y-t) - H(X-t,Y-t)
            conditional_entropy_joint = H1 - H2

            ### And Conditional Entropy independent of X(t) H(Y|Y-t) = H(Y,Y-t) - H(Y-t)            
            conditional_entropy_independent = H3 - H4

            ### Directional Transfer Entropy is the difference between the conditional entropies
            transfer_entropies[i] = conditional_entropy_independent - conditional_entropy_joint

        TEs.append(transfer_entropies)

        ## Calculate Significance of TE during this window
        if n_shuffles > 0:
            p, z, TE_mean = significance(df=df,
                                            TE=transfer_entropies,
                                            endog=self.endog,
                                            exog=self.exog,
                                            lag=self.lag,
                                            n_shuffles=n_shuffles,
                                            bandwidth=bandwidth,
                                            method='transfer_entropy')

            shuffled_TEs.append(TE_mean)
            p_values.append(p)
            z_scores.append(z)
            ## Store Significance Transfer Entropy from X(t)->Y(t) and from Y(t)->X(t)
            self.results = np.concatenate((np.array(TEs), np.array(p_values), np.array(z_scores), np.array(shuffled_TEs)), axis=0)
        else:
            ## Store Significance Transfer Entropy from X(t)->Y(t) and from Y(t)->X(t)
            self.results = np.array(TEs)
            
        return self.results
    
    def MutualInformation(self, df=None, bandwidth=None, gridpoints=20, n_shuffles=0):
        ## Prepare lists for storing results
        Norm_MIs = []
        shuffled_MIs = []
        p_values = []
        z_scores = []

        df = np.copy(self.array)

        ## Initialise list to return TEs
        # mutual_informations = [0, 0]
        normalized_mi = [0, 0]

        ## Require us to compare information transfer bidirectionally
        for i in range(2):
            ### Entropy calculated using Probability Density Estimation:
            ### Estimate PDF using Gaussian Kernels and use H(x) = p(x) log p(x)
            ## 1. H(Y,Y-t,X-t)
            H1 = self.get_entropy(df=df[:, [1 - i]],
                                gridpoints=gridpoints,
                                bandwidth=bandwidth,
                                covar=self.covars[i][0])
            ## 2. H(Y-t,X-t)
            H2 = self.get_entropy(df=df[:, [i]],
                                gridpoints=gridpoints,
                                bandwidth=bandwidth,
                                covar=self.covars[i][1])
            ## 3. H(Y,Y-t)
            H3 = self.get_entropy(df=df[:, [1 - i, i]],
                                gridpoints=gridpoints,
                                bandwidth=bandwidth,
                                covar=self.covars[i][2])

            normalized_mi[i] = 1 - (H1 + H2 - H3 / max(H1, H2))

        Norm_MIs.append(normalized_mi)

        ## Calculate Significance of TE during this window
        if n_shuffles > 0:
            p, z, MI_mean = significance(df=df,
                                            TE=normalized_mi,
                                            endog=self.endog,
                                            exog=self.exog,
                                            lag=self.lag,
                                            n_shuffles=n_shuffles,
                                            bandwidth=bandwidth,
                                            method='mutual_information')

            shuffled_MIs.append(MI_mean)
            p_values.append(p)
            z_scores.append(z)
            ## Store Significance Transfer Entropy from X(t)->Y(t) and from Y(t)->X(t)
            self.results = np.concatenate((np.array(Norm_MIs), np.array(p_values), np.array(z_scores), np.array(shuffled_MIs)), axis=0)
        else:
            ## Store Significance Transfer Entropy from X(t)->Y(t) and from Y(t)->X(t)
            self.results = np.array(Norm_MIs)
            
        return self.results

    def get_entropy(self, df, gridpoints=20, bandwidth=None, covar=None):
        """
            Function for calculating entropy from a probability mass 
            
        Args:
            df          -       (DataFrame) Samples over which to estimate density
            gridpoints  -       (int)       Number of gridpoints when integrating KDE over 
                                            the domain. Used if estimator='kernel'
            bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                            matrix). Used if estimator='kernel'
            estimator   -       (string)    'histogram' or 'kernel'
            bins        -       (Dict of lists) Bin edges for NDHistogram. Used if estimator
                                            = 'histogram'
            covar       -       (Numpy ndarray) Covariance matrix between dimensions of df. 
                                            Used if estimator = 'kernel'
        Returns:
            entropy     -       (float)     Shannon entropy in bits

        """
        # df is np.array
        pdf = self.pdf_kde(df, gridpoints, bandwidth, covar)
        # log base 2 returns H(X) in bits
        return -np.sum(pdf * ma.log2(pdf).filled(0))
    
    def pdf_kde(self, df, gridpoints=None, bandwidth=1, covar=None):
        """
            Function for non-parametric density estimation using Kernel Density Estimation

        Args:
            df          -       (DataFrame) Samples over which to estimate density
            gridpoints  -       (int)       Number of gridpoints when integrating KDE over 
                                            the domain. Used if estimator='kernel'
            bandwidth   -       (float)     Bandwidth for KDE (scalar multiple to covariance
                                            matrix).
            covar       -       (Numpy ndarray) Covariance matrix between dimensions of df. 
                                            If None, these are calculated from df during the 
                                            KDE analysis

        Returns:
            Z/Z.sum()   -       (Numpy ndarray) Probability of a sample being between
                                            specific gridpoints (technically a probability mass)
        """
        # df is np.array
        ## Create Meshgrid to capture data
        if gridpoints is None:
            gridpoints = 20
        
        N = complex(gridpoints)
        
        slices = [slice(dim.min(),dim.max(),N) for dim in df.T]
        grids = np.mgrid[slices]

        ## Pass Meshgrid to Scipy Gaussian KDE to Estimate PDF
        positions = np.vstack([X.ravel() for X in grids])
        values = df.T
        kernel = _kde_(values, bw_method=bandwidth, covar=covar)
        Z = np.reshape(kernel(positions).T, grids[0].shape) 

        ## Normalise 
        return Z/Z.sum()
    

class _kde_(stats.gaussian_kde):
    """
    Subclass of scipy.stats.gaussian_kde. This is to enable the passage of a pre-defined covariance matrix, via the
    `covar` parameter. This is handled internally within TransferEntropy class.
    The matrix is calculated on the overall dataset, before windowing, which allows for consistency between windows,
    and avoiding duplicative computational operations, compared with calculating the covariance each window.

    Functions left as much as possible identical to scipi.stats.gaussian_kde; docs available:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    """
    def __init__(self, dataset, bw_method=None, df=None, covar=None):
        self.dataset = np.atleast_2d(dataset)
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        self.d, self.n = self.dataset.shape
        self.set_bandwidth(bw_method=bw_method, covar=covar)


    def set_bandwidth(self, bw_method=None, covar=None):
        
        if bw_method is None:
            pass
        elif np.isscalar(bw_method) and not isinstance(bw_method, string_types):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
                  "or a callable."
            raise ValueError(msg)

        self._compute_covariance(covar)

    def _compute_covariance(self, covar):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        if covar is not None:
            self._data_covariance = covar
            self._data_inv_cov = linalg.inv(self._data_covariance)
        
        self.factor = self.covariance_factor()
        # Cache covariance and Cholesky decomp of covariance
        if not hasattr(self, '_data_cho_cov'):
            self._data_covariance = np.atleast_2d(np.cov(self.dataset, rowvar=1,
                                            bias=False))
            self._data_cho_cov = linalg.cholesky(self._data_covariance,
                                                lower=True)

        self.covariance = self._data_covariance * self.factor**2
        self.cho_cov = (self._data_cho_cov * self.factor).astype(np.float64)
        self.log_det = 2*np.log(np.diag(self.cho_cov
                                        * np.sqrt(2*np.pi))).sum()

    @property
    def inv_cov(self):
        # Re-compute from scratch each time because I'm not sure how this is
        # used in the wild. (Perhaps users change the `dataset`, since it's
        # not a private attribute?) `_compute_covariance` used to recalculate
        # all these, so we'll recalculate everything now that this is a
        # a property.
        self.factor = self.covariance_factor()
        self._data_covariance = np.atleast_2d(np.cov(self.dataset, rowvar=1,
                                        bias=False, aweights=self.weights))
        return linalg.inv(self._data_covariance) / self.factor**2
    

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

        if method == 'mutual information':
            ## Calculate New TE
            shuffled_causality = TransferEntropyArray(df, endog=endog, exog=exog, lag=lag, method = 'mi')
            TE_shuffled = shuffled_causality.MutualInformation(df, bandwidth, n_shuffles=0)
        else:
            ## Calculate New TE
            shuffled_causality = TransferEntropyArray(df, endog=endog, exog=exog, lag=lag)
            TE_shuffled = shuffled_causality.TransferEntropy(df, bandwidth, n_shuffles=0)
        shuffled_TEs[:, i] = TE_shuffled

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
