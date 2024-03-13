import pickle
import numpy as np

def save_pickle(var, file):
    print('SAVING PICKLE TO %s'%file)
    with open(file, 'wb') as f:
        pickle.dump(var, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        out = pickle.load(f)
    return out

def reshape_input(inp):
    return np.concatenate(inp.to_numpy()).reshape(len(inp),-1)
