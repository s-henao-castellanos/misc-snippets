from numba import vectorize, njit

def spell(signature,**kwargs):
    """Compile ahead-of-time and vectorize with the same signature in one decorator.
    The keyword arguments are passed to `numba.njit`. Numba caching is enabled."""
    def inner(function):
        return vectorize(signature)(njit(signature,**(dict(cache=True)|kwargs))(function))
    return inner