import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import optim
from ml.utils import update_params_from_cmdline, save_metrics_params, ensure_dir, ensure_empty_dir, infinite_dataset
from load_dataset_T import *
import torch
import numpy as np
from sklearn.preprocessing import minmax_scale
import torchvision.transforms as transforms
import os
from tensorboardX import SummaryWriter
from ml.pytorch_modules.vae import BVAE
from torch.utils.data import Dataset, DataLoader
from vae_many import *
latent_dims =2
device = 'cpu'
model =  VariationalAutoencoder(latent_dims).to(device)
default_parameters = {'model_dir': './results/ld_2'}
def idx(x,y):
    ix =[]
    for i in range(len(x)):
        if x[i] == y:
            ix.append(i)
    return ix


# To make plots pretty
golden_size = lambda width: (width, 2. * width / (1 + np.sqrt(5)))

root=path_to_data=os.path.expanduser('~')+'/Dropbox/IsingData/'
#Clasification
hmp =[] 
for t in np.arange(0.25,4.01,0.25):
    print(t)
    X = load_data_set(t,root= root)
    model.load_state_dict(torch.load('./trained_models/Ising_dense_%d_T_%f'%(latent_dims,t)))#%(latent_dims))) #we load the saved parameters
    x_train, x_test = train_test_split(X, test_size=0.2)
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) # flatten each sample out
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    zpred, mu, var = model.encode(torch.tensor(x_test).float())
    hmp_temp,_= np.histogram(zpred.detach().numpy(),50)
    hmp =hmp+ [hmp_temp] 

plt.rc('font',**{'size':16})
fig, ax = plt.subplots(1,figsize=golden_size(8))
plt.imshow(hmp, cmap='Greys', interpolation='nearest')
ax.set_xlabel('First latent z of the VAE')
ax.set_ylabel('Temperature')
plt.savefig('VAE_ISING_hmp_%d_dense_many.png'%(latent_dims))
