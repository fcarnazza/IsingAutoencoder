import torch; torch.manual_seed(0)
from ml.utils import update_params_from_cmdline, save_metrics_params, ensure_dir, ensure_empty_dir, infinite_dataset
from sklearn.preprocessing import minmax_scale
import os
import sys
from train_data import *
from load_dataset_traj import *
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
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
        return z#.reshape((-1, 1, 40, 40))

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
        self.kl = 0.5*(sigma**2 + mu**2 - 2*torch.log(sigma) - 1).sum()
        return z


class ConvEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)  # 32 x 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 16 x 16
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)  # 4 x 4
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 512, 4)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 1, 64, 64)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return z

class VariationalConvEncoder(nn.Module):
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
        self.kl = 0.5*(sigma**2 + mu**2 - 2*torch.log(sigma) - 1).sum()
        return z

class ConvDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 64, 4, 1, 0)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 32 x 32
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, 4, 2, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv_final(h)
        return mu_img

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
            x = torch.flatten(x, start_dim=1)
            xent_loss = f.binary_cross_entropy( x_hat,x  )
            loss = xent_loss + beta*autoencoder.encoder.kl
            loss.backward()
            summed_train_loss += loss.item()
            opt.step()
            #writer.add_scalar('train_loss', summed_train_loss, idx)
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
            loss = 0.5*((x - x_hat)**2).sum() + autoencoder.encoder.kl
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
                beta = 1#float(sys.argv[1]) #[0.1,0.2,0.5,0.7,1,2,5,20,100]:
                T = float(sys.argv[1])
                #cfg = update_params_from_cmdline(default_params=default_parameters)
                # old  root='../MHIsing/Ising_2d_traj/num_traj_1000_traj_20.00.h5'
                file_name = '../ising/Ising_2d_traj/num_traj_all_time_1000_traj_%.3f_corr.h5'%T
                #file_name = 'ising_2d_traj/num_traj_all_time_1000_traj_0.44_corr.h5'
                # old data_train = Data_p_train(root)
                # old train_loader = DataLoader(dataset=data_train, batch_size=100, shuffle=True, num_workers=0)

                # Only the first timestep to see the transistion, comment this lines and X_tot -> X
                X, Y = load_data_set(file_name)
                #X = np.array( [X_tot[i*1002:i*1002+120] for i in range(1000)]   ).reshape(120*1000,40,40)
                #Y = np.array( [Y_tot[i*1002:i*1002+120] for i in range(1000)]   ).reshape(120*1000,)
                print(X.shape)
                x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.8)
                x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) # flatten each sample out
                x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
                x_train = minmax_scale(x_train) # this step is required in order to use cross-entropy loss for reconstruction
                x_test = minmax_scale(x_test) # scaling features in 0,1 interval
                train_loader = DataLoader(dataset=torch.tensor(x_train).float(), batch_size=100, shuffle=True, num_workers=0)
                latent_dims = 10
                vae = VariationalAutoencoder(latent_dims).to(device) # GPU
                #writer = SummaryWriter(os.path.join(cfg.model_dir, 'tensorboard'))
                vae = train(vae,train_loader,beta=beta ,epochs =10)
                torch.save(vae.state_dict(), './trained_models/Ising_dense_traj_%d_beta_%.2f_T_%.3f'%(latent_dims,beta,T))
                train_test = 'T_%.3f_test_train_beta_%.2f.npy'%(T,beta)
                path = os.path.join( 'trained_models', train_test)
                with open(path, 'wb') as f:
                    np.save( f, x_test )
                    np.save( f, y_test )
                    np.save( f, x_train )
                    np.save( f, y_train )
