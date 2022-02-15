import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import optim
from ml.utils import update_params_from_cmdline, save_metrics_params, ensure_dir, ensure_empty_dir, infinite_dataset
from load_dataset_Ising_1d import *
import torch
import numpy as np
from sklearn.preprocessing import minmax_scale
import torchvision.transforms as transforms
import os
from tensorboardX import SummaryWriter
from ml.pytorch_modules.vae import BVAE
from torch.utils.data import Dataset, DataLoader
from vae_Ising_1d import *
latent_dims =2
original_dims =100
device = 'cpu'
model =  VariationalAutoencoder(original_dims,latent_dims).to(device)
default_parameters = {'model_dir': './results/ld_2'}
def idx(x,y):
    ix =[]
    for i in range(len(x)):
        if x[i] == y:
            ix.append(i)
    return ix
model.load_state_dict(torch.load('./trained_models/Ising_1d_dense_%d'%(latent_dims)))#%(latent_dims))) #we load the saved parameters
filename = 'sample_afm.h5'
X, Y = load_data_set(filename)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]) ) # flatten each sample out
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))
zpred, mu, var = model.encode(torch.tensor(x_test).float())
x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:]) )) # flatten each sample out
x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))


# To make plots pretty
golden_size = lambda width: (width, 2. * width / (1 + np.sqrt(5)))


#Clasification
hmp =[] 
for t in np.arange(0.1,2.1,0.1):
    i_temp = idx(y_test,t)
    ztemp = np.array([mu.detach().numpy()[i_temp[i]][0] for i in range(len(i_temp))])
    hmp_temp,_= np.histogram(ztemp,50)
    hmp =hmp+ [hmp_temp] 
    plt.rc('font',**{'size':16})
    fig, ax = plt.subplots(1,figsize=golden_size(8))
    ax.set_xlabel('First latent mu of the VAE')
    ax.set_ylabel('#samples encoded in first mu at T %f'%(t))
    plt.hist(ztemp,bins=50)
    plt.savefig('VAE_ISING_latent_order_mu_1d_%d_dense_T_%f.png'%(latent_dims, t))


plt.rc('font',**{'size':16})
fig, ax = plt.subplots(1,figsize=golden_size(8))
plt.imshow(hmp, cmap='Greys', interpolation='nearest')
ax.set_xlabel('First latent mu of the VAE')
ax.set_ylabel('Temperature')
plt.savefig('VAE_ISING_hmp_mu_1d_%d_dense.png'%(latent_dims))



exit()
#%matplotlib inline
plt.rc('font',**{'size':16})
fig, ax = plt.subplots(1,figsize=golden_size(8))
sc = ax.scatter(zpred[:,0], zpred[:,1], c=y_test/4.0, s=4, cmap="coolwarm")
ax.set_xlabel('First latent dimension of the VAE')
ax.set_ylabel('Second latent dimension of the VAE')
plt.colorbar(sc, label='$0.25\\times$Temperature')
plt.savefig('VAE_ISING_latent_%d_dense.png'%(latent_dims))
plt.show()



plt.rc('font',**{'size':16})
fig, ax = plt.subplots(1,2,figsize=(15,8))
ax[0].scatter(zpred[:,0], np.mean(x_test, axis=1), c=y_test/4.0, s=2, cmap="coolwarm")
ax[0].set_xlabel('First latent dimension of the VAE')
ax[0].set_ylabel('Magnetization')
sc = ax[1].scatter(zpred[:,1], np.mean(x_test, axis=1), c=y_test/4.0, s=2, cmap="coolwarm")
ax[1].set_xlabel('Second latent dimension of the VAE')
ax[1].set_ylabel('Magnetization')
plt.colorbar(sc, label='$0.25\\times$Temperature')
plt.savefig('VAE_ISING_latent_magnetization_%d_dense.png'%(latent_dims ))
plt.show()

