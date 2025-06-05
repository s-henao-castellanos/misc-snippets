#from multiprocessing import Pool
from multiprocess import Pool # better pickling with dill
from tqdm import tqdm
            

def parallel_map(f,array,processes=8):
    with Pool(processes=processes) as p:
        return list(
            tqdm(
                p.imap(f,array),
                total=len(array)
            )
        )
