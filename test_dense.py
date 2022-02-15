import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import optim
from ml.utils import update_params_from_cmdline, save_metrics_params, ensure_dir, ensure_empty_dir, infinite_dataset
from load_dataset import *
import torch
import numpy as np
from sklearn.preprocessing import minmax_scale
import torchvision.transforms as transforms
import os
from tensorboardX import SummaryWriter
from ml.pytorch_modules.vae import BVAE
from torch.utils.data import Dataset, DataLoader
from vae2 import *
latent_dims =20
device = 'cpu'
model =  VariationalAutoencoder(latent_dims).to(device)
default_parameters = {'model_dir': './results/ld_2'}
def idx(x,y):
    ix =[]
    for i in range(len(x)):
        if x[i] == y:
            ix.append(i)
    return ix
model.load_state_dict(torch.load('./trained_models/Ising_dense_%d_beta_5.00'%(latent_dims)))#%(latent_dims))) #we load the saved parameters
root=path_to_data=os.path.expanduser('~')+'/Dropbox/IsingData/'
X, Y = load_data_set(root= root)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]) ) # flatten each sample out
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))
zpred, mu, sigma = model.encode(torch.tensor(x_test).float())
x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:]) )) # flatten each sample out
x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))
mu = mu.detach().numpy()
sigma = sigma.detach().numpy()
# To make plots pretty
golden_size = lambda width: (width, 2. * width / (1 + np.sqrt(5)))

#mu_var = [np.var([mu.detach().numpy()[d]]) for d in range(latent_dims)]
for i in range(latent_dims):
    plt.rc('font',**{'size':16})
    fig, ax = plt.subplots(1,figsize=golden_size(8))
    ax.set_xlabel('%d latent z of the VAE'%(i+1))
    ax.set_ylabel('#samples encoded as z')
    plt.hist(mu[:,i],bins=50)
    print(np.std(mu[:,i]))
    plt.savefig('VAE_ISING_latent_%d_order_%d_dense__beta_5.00.png'%(i+1,latent_dims))
std_of_mu=[np.std(mu[:,i]) for i in range(latent_dims) ]
mean_of_sigmas=[np.mean(sigma[:,i]) for i in range(latent_dims) ]
mean_of_sigmas2=[np.mean(sigma[:,i]**2) for i in range(latent_dims) ]

x_latent = [d for d in range(latent_dims)]
plt.rc('font',**{'size':16})
fig, ax = plt.subplots(1,figsize=golden_size(8))
ax.set_ylabel('std of mu')
ax.set_xlabel('latent dimension')
sc = plt.plot(x_latent,std_of_mu)
plt.savefig('std_of_mu_%d_dense__beta_5.00.png'%(latent_dims))

plt.rc('font',**{'size':16})
fig, ax = plt.subplots(1,figsize=golden_size(8))
ax.set_ylabel('mean of sigma')
ax.set_xlabel('latent dimension')
sc = plt.plot(x_latent,mean_of_sigmas )
plt.savefig('mean_of_sigmas_%d_dense__beta_5.00.png'%(latent_dims))

#Clasification
plt.rc('font',**{'size':16})
fig, ax = plt.subplots(1,figsize=golden_size(8))
ax.set_ylabel('mean of sigma^2')
ax.set_xlabel('latent dimension')
sc = plt.plot(x_latent,mean_of_sigmas2 )
plt.savefig('mean_of_sigmas2_%d_dense__beta_5.00.png'%(latent_dims))

hmp =[] 
for t in np.arange(0.25,4.01,0.25):
    i_temp = idx(y_test,t)
    ztemp = np.array([mu[i_temp[i]][0] for i in range(len(i_temp))])
    hmp_temp,_= np.histogram(ztemp,50)
    hmp =hmp+ [hmp_temp] 
    plt.rc('font',**{'size':16})
    fig, ax = plt.subplots(1,figsize=golden_size(8))
    ax.set_xlabel('First latent mu of the VAE')
    ax.set_ylabel('#samples encoded in first mu at T %f'%(t))
    plt.hist(ztemp,bins=50)
    plt.savefig('./images/VAE_ISING_latent_order_mu_%d_dense__beta_5.00_T_%f.png'%(latent_dims, t))


plt.rc('font',**{'size':16})
fig, ax = plt.subplots(1,figsize=golden_size(8))
plt.imshow(hmp, cmap='Greys', interpolation='nearest')
ax.set_xlabel('First latent mu of the VAE')
ax.set_ylabel('Temperature')
ax.set_yticks(np.arange(0.25,4.01,0.25))
plt.savefig('./images/VAE_ISING_hmp_mu_%d_dense__beta_5.00.png'%(latent_dims))



exit()
#%matplotlib inline
plt.rc('font',**{'size':16})
fig, ax = plt.subplots(1,figsize=golden_size(8))
sc = ax.scatter(zpred[:,0], zpred[:,1], c=y_test/4.0, s=4, cmap="coolwarm")
ax.set_xlabel('First latent dimension of the VAE')
ax.set_ylabel('Second latent dimension of the VAE')
plt.colorbar(sc, label='$0.25\\times$Temperature')
plt.savefig('VAE_ISING_latent_%d_dense__beta_5.00.png'%(latent_dims))
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
plt.savefig('VAE_ISING_latent_magnetization_%d_dense__beta_5.00.png'%(latent_dims ))
plt.show()

