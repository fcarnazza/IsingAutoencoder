import pickle
from sklearn.model_selection import train_test_split
import collections
import numpy as np
def load_data_set(root="IsingMC/", train_size = 0.5):
    """Loads the Ising dataset in the format required for training the tensorflow VAE
    
    Parameters
    -------
    
    root: str, default = "IsingMC/"
        Location of the directory containing the Ising dataset
    train_size: float, default = 0.5
        Size ratio of the training set. 1-train_size corresponds to the test set size ratio.
        
    """
    # The Ising dataset contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
    data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=All.pkl','rb'))
    data = np.unpackbits(data).astype(float).reshape(-1,1600) # decompression of data and casting to int. 
    Y = np.hstack([t]*10000 for t in np.arange(0.25,4.01,0.25)) # labels

    # Here we downsample the dataset and use 1000 samples at each temperature
    tmp = np.arange(10000)
    np.random.shuffle(tmp)
    rand_idx=tmp[:10000]
    
    X = np.vstack(data[i*10000:(i+1)*10000][rand_idx] for i, _ in enumerate(np.arange(0.25,4.01,0.25)))
    Y = np.hstack(Y[i*10000:(i+1)*10000][rand_idx] for i, _ in enumerate(np.arange(0.25,4.01,0.25)))
    # Note that data is not currently shuffled
    
    return X, Y
