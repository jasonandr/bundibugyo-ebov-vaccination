import generate_final_outputs as gfo
from functools import partial
import multiprocessing

if __name__ == '__main__':
    gfo.n_reps = 1
    pool = multiprocessing.Pool(1)
    results_no_int = pool.map(partial(gfo.run_rep, gfo.G, gfo.scenarios["No Intervention"]), range(1))
    res = results_no_int[0]
    print(type(res))
    if isinstance(res, dict):
        print(res.keys())
    else:
        print(res)
