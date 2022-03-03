import torch; torch.manual_seed(0)
from ml.utils import update_params_from_cmdline, save_metrics_params, ensure_dir, ensure_empty_dir, infinite_dataset
from sklearn.preprocessing import minmax_scale
import os
from load_dataset import *
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
default_parameters = {'model_dir': './results/ld_2'}
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 1600)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z#.reshape((-1, 1, 28, 28))

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(1600, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
#        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
#        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def encode(self,x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        return z, mu, sigma

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(1600, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def encode(self,x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        return z, mu, sigma

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z



class ConvDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 64, 5, 1, 0)  # 5 x 5
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 10 x 10
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 20 x 20
        self.bn4 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, 4, 2, 1)
        self.input_dim = input_dim
        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)
    def decode(self,z):
        h = z.view( 1,z.size(0), 1, 1)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        mu_img = self.conv_final(h)
        return mu_img.view( mu_img.size(2),mu_img.size(3) )

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        mu_img = self.conv_final(h)
        return mu_img.view(mu_img.size(0),mu_img.size(1),mu_img.size(2)*mu_img.size(3))

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)
    def decode(self,z):
        return self.decoder(z)

    def encode(self, x):
        return self.encoder.encode(x)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class VariationalDeconvAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalDeconvAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = ConvDecoder(latent_dims)
    def decode(self,z):
        return self.decoder.decode(z)

    def encode(self, x):
        return self.encoder.encode(x)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train(autoencoder, data, beta=1,epochs=5):
    opt = torch.optim.Adam(autoencoder.parameters())
    idx = 0
    for epoch in range(epochs):
        summed_train_loss = 0.
        for x in data:
            idx +=1 
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat.view(x.size()))**2).sum() + beta*autoencoder.encoder.kl
            loss.backward()
            summed_train_loss += loss.item()
            opt.step()
            writer.add_scalar('train_loss', summed_train_loss, idx)
        print('=== Mean train loss: {:.12f}'.format(summed_train_loss / 100))
    return autoencoder


def eval(autoencoder, data, epochs=5):
    autoencoder.eval()
    opt = torch.optim.Adam(autoencoder.parameters())
    idx = 0
    summed_eval_loss = 0.
    for epoch in range(epochs):
        for x in data:
            idx +=1 
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            summed_train_loss += loss.item()
            opt.step()
            writer.add_scalar('train_loss', summed_train_loss, idx)
    return autoencoder

def generate_samples(model, z_input,n=1):
    temp=model.decode(z_input).detach().numpy().reshape(n*n,1600)
    draws=np.random.uniform(size=temp.shape)
    samples=np.array(draws<temp).astype(int)
    return samples, temp


if __name__ == '__main__':
    for beta in [0.1,0.2,0.5,0.7,1,2,5,20,100]:
        cfg = update_params_from_cmdline(default_params=default_parameters)
        root = './IsingData/' #path_to_data=os.path.expanduser('~')+'/Dropbox/IsingData/'
        X, Y = load_data_set(root= root)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.8)
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) # flatten each sample out
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        x_train = minmax_scale(x_train) # this step is required in order to use cross-entropy loss for reconstruction
        x_test = minmax_scale(x_test) # scaling features in 0,1 interval
        train_loader = DataLoader(dataset=torch.tensor(x_train).float(), batch_size=100, shuffle=True, num_workers=0)
        writer = SummaryWriter(os.path.join(cfg.model_dir, 'tensorboard'))

        latent_dims = 10
        vae = VariationalDeconvAutoencoder(latent_dims).to(device) # GPU
        vae = train(vae,train_loader,beta=beta )
        torch.save(vae.state_dict(), './trained_models/Ising_dc_%d_beta_%.2f'%(latent_dims,beta))










