import pickle
from sklearn.model_selection import train_test_split
import collections
import numpy as np
def load_data_set(T,root="IsingMC/", train_size = 0.5):
    """Loads the Ising dataset in the format required for training the tensorflow VAE
    
    Parameters
    -------
    
    root: str, default = "IsingMC/"
        Location of the directory containing the Ising dataset
    train_size: float, default = 0.5
        Size ratio of the training set. 1-train_size corresponds to the test set size ratio.
        
    """
    # The Ising dataset contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
    print( root+'Ising2DFM_reSample_L40_T=%.2f.pkl'%(T)  )
    data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=%.2f.pkl'%(T),'rb'))
    data = np.unpackbits(data).astype(float).reshape(-1,1600) # decompression of data and casting to int. 
    print(data)    
    return data 



