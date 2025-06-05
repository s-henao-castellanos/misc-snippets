import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from joblib import Parallel, delayed
import toolz as tz
from astropy.io import fits
from tqdm.notebook import tqdm
from glob import glob
import h5py
import re

listmap = tz.curry(tz.compose(list,map))
array_concat = tz.curry(np.concat)


np.set_printoptions(linewidth=120)

plt.rcParams["image.cmap"] = "magma"

class ArrayCover:
    def __init__(self,array):
        self.array = array
    def __repr__(self):
        return f"<ArrayCover {self.array.shape}>"


@tz.curry
def map_at(func,level,arg):
    "Applies the funcion `func` only at the requested level of object `arg`. For instance, level 2 will apply the function to every cell in a matrix."
    return tz.compose(*([listmap]*level))(func)(arg)

@tz.curry
def array_split_multi_axis(A,ns,start_axis=0):
    """Splits the i-th axis of the array A in ns[i] chunks. 
    The resulting object is a nested list with the same number of dimensions as the length of n, 
    and each element is an numpy.array representing the uncut final pieces. """
    if not ns:
        return A
    else:
        return listmap(array_split_multi_axis(ns=ns[1:],start_axis=start_axis+1),np.array_split(A,ns[0],axis=start_axis))



def covered_array_split_multi_axis(A,ns,start_axis=0):
    """Splits the i-th axis of the array A in ns[i] chunks. 
    The resulting object is a numpy array with the same number of dimensions as the length of n, 
    but each element is an ArrayCover representing the uncut final pieces. """
    return np.array(map_at(ArrayCover,len(ns),array_split_multi_axis(A,ns,start_axis)))


def nested_list_depth(nl):
    "Finds the number of ways you can do `nl[0][0]...[0]`"
    if isinstance(nl,list):
        return 1+nested_list_depth(nl[0])
    else:
        return 0

def array_join_multi_axis(NL):
    """The inverse function of `array_split_multi_axis`. 
    This function joins (np.concat) the most nested elements of the structure until only one array is left.
    Example:
    >>> n = 10
    >>> A = np.arange(n**3).reshape(n,n,n)
    >>> test = array_split_multi_axis(A,[3,2,4])
    >>> np.array_equal(array_join_multi_axis(test),A)
    True"""
    depth = nested_list_depth(NL)
    foos = [map_at(array_concat(axis=-i),depth-i) for i in range(1,depth+1)]
    return tz.compose_left(*foos)(NL)

def covered_array_join_multi_axis(array_of_covers):
    depth = nested_list_depth(array_of_covers.tolist())
    NL = map_at(lambda x: x.array, depth, array_of_covers)
    return array_join_multi_axis(NL)


def array_chunked_map(func,A,ns):
    """Applies the function `func` to the multidimensional array `A` by separating it into chunks defined by `ns`. 
    The i-th axis of `A` is split into `ns[i]` parts of possibly uneven (albeit balanced) size.
    The function `func` must return an array. The result is the same as applying the function to A only if the reduction axis is not splitted."""
    chunks = covered_array_split_multi_axis(A,ns)
    raveled = listmap(func,chunks.ravel())
    processed_chunks = np.reshape(raveled,chunks.shape)
    return covered_array_join_multi_axis(processed_chunks)



