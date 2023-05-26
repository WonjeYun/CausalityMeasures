from numba.pycc import CC
import numpy as np

cc = CC('ols_gc')

@cc.export('ols_res_cal', 'f8[:](f8[:],f8[:,:])')
def ols_res_cal(y, x):
    x = np.append(np.ones((len(x), 1)), x, axis=1)
    ins = np.linalg.inv(np.dot(x.T, x))
    out = np.dot(x.T, y)
    beta = np.dot(ins, out)
    res = y - np.dot(x, beta)
    return res

@cc.export('granger_cal', 'f8(f8[:],f8[:])')
def granger_cal(independent_residuals, joint_residuals):
    ind_res_var = np.var(independent_residuals) + np.finfo(np.float64).eps
    jnt_res_var = np.var(joint_residuals) + np.finfo(np.float64).eps
    gc =  np.log(ind_res_var / jnt_res_var)
    return gc

if __name__ == "__main__":
    cc.compile()