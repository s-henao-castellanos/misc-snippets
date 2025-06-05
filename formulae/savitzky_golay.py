import numpy as np

@np.vectorize(otypes=(float,float),excluded=[1,2,3,4])
def savitzky_golay_filter(x0,x,y,window,deg):
    """Performs a local regression of the data `x,y` to a polynomial of the given degree `deg`, 
    only considering the points that are at a 1-cyclic distance window/2 from `x0`"""
    mask = abs(center_mod(x-x0,1)) < window/2
    p = np.poly1d(np.polyfit(center_mod(x[mask]-x0,1)+x0,y[mask],deg=deg))
    return p(x0),p.deriv()(x0)
