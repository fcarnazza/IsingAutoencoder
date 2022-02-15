import pickle
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import collections

def load_data_set(filename,train_size=0.5):
    """Loads the Ising dataset in the format required for training the tensorflow VAE
    
    Parameters
    -------
    
    root: str, default = "IsingMC/"
        Location of the directory containing the Ising dataset
    train_size: float, default = 0.5
        Size ratio of the training set. 1-train_size corresponds to the test set size ratio.
        
    """
    data =[]
    with h5py.File(filename, "r") as f:
    # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = np.array(list(f[a_group_key]))
    # The Ising dataset contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
    #data = pickle.load(open(root+'Ising2DFM_reSample_L40_T=All.pkl','rb'))
    #data = np.unpackbits(data).astype(int).reshape(-1,1600) # decompression of data and casting to int. 
    data = data[100:].reshape(20000,100)
    print(data.shape)
    Y = np.hstack([t]*1000 for t in np.arange(0.1,2.1,0.1)) # labels

    # Here we downsample the dataset and use 1000 samples at each temperature
    tmp = np.arange(1000)
    np.random.shuffle(tmp)
    rand_idx=tmp[:1000]
    
    X = np.vstack(data[i*1000:(i+1)*1000][rand_idx] for i, _ in enumerate(np.arange(0.1,2.1,0.1)))
    Y = np.hstack(Y[i*1000:(i+1)*1000][rand_idx] for i, _ in enumerate(np.arange(0.1,2.1,0.1)))
    # Note that data is not currently shuffled
    
    return X, Y
