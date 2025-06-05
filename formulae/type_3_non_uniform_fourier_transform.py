import numpy as np
from numba import njit

@njit("f8[::](f8[::],f8[::],f8,f8,f8)",cache=True)
def NUDFT_grid(t_arg,m_arg,f_ini,f_end,df):
    """Non-Uniform Discrete Fourier Transform on an even frequency grid 
    with the Kurtz (1985, MNRAS 213,773) algorithm"""
    dF = np.exp(-2j*np.pi*t_arg*df)
    F = np.zeros_like(np.arange(f_ini,f_end,df))
    curl = m_arg * np.exp(-2j*np.pi*t_arg*f_ini)
    F[0] = np.abs(np.sum(curl))**2
    for i in range(1,len(F)):
        curl = curl*dF
        F[i] = np.abs(np.sum(curl))**2
    return F

@njit("f8(f8[:],f8[:],f8)",cache=True)
def NUDFT(t,m,f):
    "Non-Uniform Discrete Fourier Transform at a single frequency point"
    curl = m*np.exp(-2j*np.pi*t*f)
    return np.abs(np.sum(curl))**2