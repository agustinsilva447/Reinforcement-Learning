import pickle                       # to save/load Q-Tables
import numpy as np                  # for array stuff and random

start_q_table = None        # if we have a pickled Q table, we'll put the filename of it here.
N_SIZE = 3
angulos = np.arange(0, 2 * np.pi, 2 * np.pi / np.power(2, N_SIZE))

if start_q_table is None:
    # initialize the q-table #
    q_table = {}
    for rx in angulos:
        for ry in angulos:
            for rz in angulos:
                q_table[(rx,ry,rz)] = [np.random.uniform(50, 100) for i in range(2)]     
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

print(q_table)