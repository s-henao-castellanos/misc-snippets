# Stochatic Runge-Kutta second weak order 

import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit
def simulate_Ornstein_Uhlenbeck(n_iter,dt,σ,θ,x0):
    sdt = np.sqrt(dt)
    W = np.random.normal(0,1,n_iter)
    S = np.random.choice(np.array([-1,1]),size=n_iter)
    U = np.zeros(n_iter)
    U[-1] = x0
    for i in range(n_iter):
        # notice the change in sign of S
        K1 = sdt*(W[i]-S[i])*σ - dt*θ*U[i-1]
        K2 = sdt*(W[i]+S[i])*σ - dt*θ*(U[i-1]+K1)
        U[i] = U[i-1]+ dt * 1/2 * (K1+K2)
    return U

