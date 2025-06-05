# MODIFIED
#@spell("f8(f8)")
def intensity(x):#,dx,weighted=True):
    """Assumes that the data is a collection of normally distributed uncorrelated random variables with means x[i] and variances dx[i]**2.
    Converts the magnitudes to fluxes, performs a mean on the fluxes, and converts the result back to magnitudes.
    It weighted is True, the weights are assume to be 1/dx**2. Returns a tuple (result,error)."""
    y = 10**(-2/5*x)
    y_bar = y.mean()
    z = 5/2 * np.log10(1/y_bar)
    return z