import torch; torch.manual_seed(0)
from ml.utils import update_params_from_cmdline, save_metrics_params, ensure_dir, ensure_empty_dir, infinite_dataset
from sklearn.preprocessing import minmax_scale
import os
from load_dataset_T import *
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from tensorboardX import SummaryWriter
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
default_parameters = {'model_dir': './results/conv_convae_20'}
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 1600)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 40, 40))

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.cnn1 = CEncoder( latent_dims)
        self.cnn2 = CEncoder( latent_dims)

        self.N = torch.distributions.Normal(0, 1)
#        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
#        self.N.scale = self.N.scale.cuda()
        self.kl = 0
    def encode(self, x):
        mu =  self.cnn1(x)
        sigma = torch.exp(self.cnn2(x))
        z = mu + sigma*self.N.sample(mu.shape)
        return z, mu, sigma
    
    def forward(self, x):
        mu =  self.cnn1(x)
        sigma = torch.exp(self.cnn2(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = CDecoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def decode(self,z):
        return self.decoder(z)

    def encode(self, x):
        return self.encoder.encode(x)

def train(autoencoder, data, epochs=5):
    opt = torch.optim.Adam(autoencoder.parameters())
    idx = 0
    for epoch in range(epochs):
        summed_train_loss = 0.
        for x in data:
            idx +=1 
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            summed_train_loss += loss.item()
            opt.step()
            writer.add_scalar('train_loss',loss, idx)
        print('=== Mean train loss: {:.12f}'.format(summed_train_loss / 100))
    return autoencoder

# define the CNN architecture

class CEncoder(nn.Module):
    def __init__(self,latent_dims):
        super(CEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 2 * 2, 100)
        self.fc2 = nn.Linear(100, latent_dims)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 2 * 2)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x



class CDecoder(nn.Module):
    def __init__(self,latent_dims):
        super(CDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(1, 16, 6)
        self.conv2 = nn.ConvTranspose2d(16, 32,6 )
        self.conv3 = nn.ConvTranspose2d(32, 1, 6)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear( latent_dims,100)
        self.fc2 = nn.Linear( 115*115,1600)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        # add sequence of convolutional and max pooling layers
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv1(x))
        x = x.view((-1,1,115**5))
        x = self.fc2(x)
        x = self.dropout(x)
        return x.reshape((-1, 1, 40, 40))
if __name__ == '__main__': 
    T = 0.25
    cfg = update_params_from_cmdline(default_params=default_parameters)
    root=path_to_data=os.path.expanduser('~')+'/Dropbox/IsingData/'
    X = load_data_set(0.25,root= root)
    x_train, x_test= train_test_split(X, test_size=0.8)
    x_train = x_train.reshape((len(x_train),1,40,40 ))#np.prod(x_train.shape[1:]) )) # flatten each sample out
    x_test = x_test.reshape((len(x_test),1,40,40 ))#np.prod(x_test.shape[1:])))
    print(x_train.shape)
#    x_train = minmax_scale(x_train) # this step is required in order to use cross-entropy loss for reconstruction
#    x_test = minmax_scale(x_test) # scaling features in 0,1 interval
    train_loader = DataLoader(dataset=torch.tensor(x_train).float(), batch_size=100, shuffle=True, num_workers=0)
#    train_loader = infinite_dataset(train_loader)
#    eval_loader = DataLoader(dataset=x_test, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    latent_dims = 20
    writer = SummaryWriter(os.path.join(cfg.model_dir, 'tensorboard'))
    vae = VariationalAutoencoder(latent_dims).to(device) # GPU
    vae = train(vae,train_loader )
    torch.save(vae.state_dict(), './trained_models/Ising_conv_%d_%d'%(latent_dims,0.25))

