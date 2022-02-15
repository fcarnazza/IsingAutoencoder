import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import optim
from ml.utils import update_params_from_cmdline, save_metrics_params, ensure_dir, ensure_empty_dir, infinite_dataset
from load_dataset import *
import torch
import numpy as np
import random
from sklearn.preprocessing import minmax_scale
import torchvision.transforms as transforms
import os
from tensorboardX import SummaryWriter
from ml.pytorch_modules.vae import BVAE
from torch.utils.data import Dataset, DataLoader
#from vae2 import *
from c_vae import *
latent_dims =20
device = 'cpu'
model =  VariationalAutoencoder(latent_dims).to(device)

model.load_state_dict(torch.load('./trained_models/Ising_conv_%d'%(latent_dims))) #we load the saved parameters

root=path_to_data=os.path.expanduser('~')+'/Dropbox/IsingData/'
X, Y = load_data_set(root= root)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
x_train = x_train.reshape((len(x_train),1,40,40 ))#np.prod(x_train.shape[1:]) )) # flatten each sample out
x_test = x_test.reshape((len(x_test),1,40,40 ))#np.prod(x_test.shape[1:])))

zpred, mu, var = model.encode(torch.tensor(x_test).float())

def generate_samples(model, z_input,n=1):
    temp=model.decode(z_input).detach().numpy().reshape(n*n,1600)
    draws=np.random.uniform(size=temp.shape)
    samples=np.array(draws<temp).astype(int)
    return samples


fig = plt.figure(figsize=(6, 2))
columns = 6
rows = 2
x_test = torch.tensor( x_test ).float()
img_rows=40
img_cols=40


for i in np.arange(1, 6):
    m = random.randint(0,32000)
    x = x_test[m].detach().numpy().reshape(40,40)
    fig.add_subplot(rows, columns, i)
    plt.imshow(x, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
    plt.axis('off')


    fig.add_subplot(rows, columns, i+6)
    z_input = zpred[m]
    #model.encode(x_test[m].reshape(1,40,10)).reshape(1,latent_dims)
    y = generate_samples(model,z_input,1) \
                     .reshape( img_rows, img_cols)
    plt.imshow(y, cmap='hot', interpolation='nearest')

    plt.axis('off')

golden_size = lambda width: (width, 2. * width / (1 + np.sqrt(5)))

plt.savefig('Example_reconstructed_conv1.pdf')
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))


mu = mu.detach().numpy().reshape(latent_dims,32000)
var = var.detach().numpy().reshape(latent_dims,32000)
mu_var = [np.var(m) for m in mu]
var_mu = [np.mean(v) for v in var]
zpred = zpred.detach().numpy() 
print("Varinaces of the mus")
print(mu_var)

print("Averages of the vars")
print(var_mu)

x_latent = [i for i in range(latent_dims)]
print("Varinaces of the mus")
plt.rc('font',**{'size':16})
fig, ax = plt.subplots(1,figsize=golden_size(8))
sc = plt.plot(x_latent,mu_var)
plt.show
plt.savefig('var_of_Mus_vae_latent_%d_conv.png'%(latent_dims))
print("Averages of the vars")
plt.rc('font',**{'size':16})
fig, ax = plt.subplots(1,figsize=golden_size(8))
sc = plt.plot(x_latent,var_mu )
plt.show
plt.savefig('mus_of_var_vae_latent_%d_conv.png'%(latent_dims))






#%matplotlib inline
plt.rc('font',**{'size':16})
fig, ax = plt.subplots(1,figsize=golden_size(8))
sc = ax.scatter(zpred[:,0], zpred[:,1], c=y_test/4.0, s=4, cmap="coolwarm")
ax.set_xlabel('First latent dimension of the VAE')
ax.set_ylabel('Second latent dimension of the VAE')
plt.colorbar(sc, label='$0.25\\times$Temperature')
plt.savefig('VAE_ISING_latent_%d_conv.png'%(latent_dims))
#plt.show()




plt.rc('font',**{'size':16})
fig, ax = plt.subplots(1,2,figsize=(15,8))
ax[0].scatter(zpred[:,0], np.mean(x_test, axis=1), c=y_test/4.0, s=2, cmap="coolwarm")
ax[0].set_xlabel('First latent dimension of the VAE')
ax[0].set_ylabel('Magnetization')
sc = ax[1].scatter(zpred[:,1], np.mean(x_test, axis=1), c=y_test/4.0, s=2, cmap="coolwarm")
ax[1].set_xlabel('Second latent dimension of the VAE')
ax[1].set_ylabel('Magnetization')
plt.colorbar(sc, label='$0.25\\times$Temperature')
plt.savefig('VAE_ISING_latent_magnetization_%d_conv.png'%(latent_dims ))
#plt.show()


#Clasification

for i in range(latent_dims):
    plt.rc('font',**{'size':16})
    fig, ax = plt.subplots(1,figsize=golden_size(8))
    ax.set_xlabel('%d latent z of the VAE'%(i+1))
    ax.set_ylabel('#samples encoded as z')
    plt.hist(zpred[:,i],bins=50)
    plt.savefig('VAE_ISING_latent_%d_order_%d_dense.png'%(i+1,latent_dims))
    plt.show()






# display a 2D manifold of the images
quantile_min = 0.01
quantile_max = 0.99

img_rows=40
img_cols=40

z_input = np.array \
        ([np.random.rand() for i in range(latent_dims)]).reshape(1,latent_dims)
z_input = torch.tensor(z_input).float()
x_pred_grid = generate_samples(model,z_input,1)
x_pred_grid = x_pred_grid.reshape( img_rows, img_cols)


    
fig, ax = plt.subplots(figsize=golden_size(10))

ax.imshow(np.block(list(map(list, x_pred_grid))), cmap='gray_r', vmin=0, vmax=1)

ax.set_xticks(np.arange(0, 1*img_rows, img_rows) + .5 * img_rows)

ax.set_yticks(np.arange(0, 1*img_cols, img_cols) + .5 * img_cols)
ax.grid(False)
plt.savefig('VAE_ISING_fantasy_invCDF.pdf')


