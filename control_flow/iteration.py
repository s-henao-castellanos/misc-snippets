from toolz import take

def prefix_idempotent(L:list[str],prefix:str) -> list[str]:
    "Adds a prefix to every element on list only if the prefix is not present already."
    return [(prefix+l if not l.startswith(prefix) else l)  for l in L]


def get_cuts(sequence,cut_list):
    """given an iterable sequence and a list of cuts, say [a,b,c,...], 
    returns a list with the first a elements, the next b, the next c, and so on"""
    tmp = iter(sequence)
    return [[*take(cut,tmp)] for cut in cut_list]