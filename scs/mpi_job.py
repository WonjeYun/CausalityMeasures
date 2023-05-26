from mpi4py import MPI
import pandas as pd
import numpy as np
from itertools import combinations
from Entropy_Causality import TransferEntropyArray
from Granger_Causality import GrangerCausality
from lag_data_gen import LaggedTimeSeriesArray

def transfer_ent(lts_array, endog, exog, t):
    """
    Calculate TE using KDE
    Creates calculation results of TE into a list as [X, Y, TE_XY, p_value_XY, z_score_XY, Ave_TE_XY]
    Two lists are created with {{exog:X, endog:Y}, {exog:Y, endog:X}}
    """
    te = TransferEntropyArray(lts_array, endog=endog, exog=exog, lag=t)
    te_res = te.TransferEntropy(n_shuffles=20)
    result = te_res.transpose()

    return result

def granger(lts_array, endog, exog, t):
    """
    Calculate TE using granger causality
    Creates calculation results of TE into a list as [X, Y, GC_XY, p_value_XY, z_score_XY, Ave_GC_XY]
    Two lists are created with {{exog:X, endog:Y}, {exog:Y, endog:X}}
    """
    gc = GrangerCausality(lts_array, endog=endog, exog=exog, lag=t)
    gc_res = gc.Granger_Caus(n_shuffles=20)
    result = gc_res.transpose()

    return result

def mutual_inf(lts_array, endog, exog, t):
    """
    Calculate Mutual Information using KDE
    Creates calculation results of MI into a list as [X, Y, Norm_MI_XY, p_value_XY, z_score_XY, Ave_MI_XY]
    Two lists are created with {{exog:X, endog:Y}, {exog:Y, endog:X}}
    """
    mi = TransferEntropyArray(lts_array, endog=endog, exog=exog, lag=t, method='mi')
    mi_res = mi.MutualInformation(n_shuffles=20)
    result = mi_res.transpose()

    return result

def mpi_work(data_name, data, comb_list, comb_bi_name, lag, method):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    cals = len(data_comb_list)
    N = int(cals / size)
    
    recvbuf = np.zeros((N, data_comb_list.shape[1], 2),dtype=float)
    # scatter two-col combination data to each process
    # each process would have a list of two-col combination data
    comm.Scatter(data_comb_list, recvbuf, root=0)
    
    calculation = np.empty((N*2, 4), dtype='float')

    for comb_no in range(N):
        reduced_data = recvbuf[comb_no]
        X = comb_list[comb_no][0]
        Y = comb_list[comb_no][1]

        lts_array = LaggedTimeSeriesArray(reduced_data, lag=lag).df
        
        # returns 2x4 array except for mutual information
        if method == 'transfer_ent':
            result_list = transfer_ent(lts_array, X, Y, lag)
        elif method == 'granger':
            result_list = granger(lts_array, X, Y, lag)
        else:
            result_list = mutual_inf(lts_array, X, Y, lag)

        calculation[comb_no * 2: (comb_no+1) * 2, :] = result_list
    
    recv_list = None
    if rank == 0:
        recv_list = np.empty((size, N*2, 4), dtype='float')
    comm.Gather(calculation, recv_list, root=0)

    if rank == 0:
        recv_list = np.reshape(recv_list, (size*N*2, 4))
        res = np.concatenate((comb_bi_name, recv_list), axis=1)

        if method == 'transfer_ent':
            result_df = pd.DataFrame(res, columns=['endog', 'exog', 'TE_XY',
                                                       'p_value_XY', 'z_score_XY', 'Ave_TE_XY'])
            result_df['ETE_XY'] = result_df['TE_XY'] - result_df['Ave_TE_XY']
        elif method == 'granger':
            result_df = pd.DataFrame(res, columns=['endog', 'exog', 'GC_XY',
                                                       'p_value_XY', 'z_score_XY', 'Ave_GC_XY'])
        else:
            result_df = pd.DataFrame(res, columns=['endog', 'exog', 'Norm_MI_XY',
                                                       'p_value_XY', 'z_score_XY', 'Ave_MI_XY'])

        result_df.to_csv(f'./data/{method}_{data_name}_mpi_result.csv', index=False)
    
    return

def gen_comb_bi_name(comb_list):
    comb_list = np.array(comb_list)
    comb_bi_name = np.empty((comb_list.shape[0]*2, comb_list.shape[1]), dtype='<U2')
    for i in range(comb_list.shape[0]):
      comb_bi_name[i*2] = comb_list[i]
      comb_bi_name[i*2 + 1] = np.flip(comb_list,1)[i]
    return comb_bi_name

if __name__ == "__main__":
    data_name = 'test_data'
    data = pd.read_csv(f'{data_name}.csv').iloc[:,1:]
    comb_list = [combs for combs in combinations(data.columns, 2)]
    comb_bi_name = gen_comb_bi_name(comb_list)
    data_comb_list = np.array([data[[comb_list[i][0], comb_list[i][1]]].to_numpy() for i in range(len(comb_list))])
    lag = 1
    method = 'granger'
    mpi_work(data_name, data, comb_list, comb_bi_name, data_comb_list, lag, method)



