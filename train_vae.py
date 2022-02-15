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




default_parameters = {'model_dir': './results/test1',
                      'batch_size': 100,
                      'batches_per_epoch': 1280,
                      'n_epochs': 5,
                      'device': 'cpu',
                      'BVAE' : {
                            'beta':1, 
                            'encoder_str': 'mlp', 
                            'decoder_str': 'mlp', 
                            'latent_size': 20, 
                            'encoder_params':{
                          	    'data_dim': 1600,
                          	    'layers': [256 ],
                                    'nonlin': 'relu',
                                    'output_nonlin': 'relu'
                                }, 
                            'decoder_params':{
                          	    'data_dim': 1600,
                          	    'layers': [256 ],
                                    'nonlin': 'relu',
                                    'output_nonlin': 'relu'
                                },			    
                            },
                      }

def train(model, optimizer, data_loader, batches_per_epoch, epoch, writer):
    model.train()
    summed_train_loss = 0.
 #   batch_idx = 0
    #for batch_idx in range(batches_per_epoch):
    for batch_in in data_loader:
        #batch_in = next(data_loader)
        data_in = batch_in.float().to(cfg.device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(data_in)
        loss = model.loss(data_in, recon_x, mu, logvar)
        summed_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        writer.add_scalar('train_loss', loss, batches_per_epoch * epoch + batch_idx)

    print('=== Mean train loss: {:.12f}'.format(summed_train_loss / batches_per_epoch))
    return model, optimizer



def eval(data_loader, model, batch_idx, writer):
    model.eval()
#    batch_idx = 0
    summed_eval_loss = 0.
    summed_eval_test_loss = 0.
    batches_per_eval = 320
    #for i in range(batches_per_eval):
    for batch_in in data_loader:
     #   batch_in = next(data_loader)
#        batch_idx += 1
        data_in = batch_in.float().to(cfg.device)
        recon_x, mu, logvar = model(data_in)
        eval_loss = model.loss(data_in, recon_x, mu, logvar) 
        summed_eval_loss += eval_loss

    writer.add_scalar('eval_loss', summed_eval_loss / batches_per_eval, batch_idx)
    print('=== Test set loss: {:.12f}'.format(summed_eval_loss / batches_per_eval))
    return {'loss': summed_eval_loss}




if __name__ == '__main__':
# prepare data loaders
# The y labels are the temperatures in np.arange(0.25,4.01,0.2) at which X was drawn

#Directory where data is stored
            cfg = update_params_from_cmdline(default_params=default_parameters)
            root=path_to_data=os.path.expanduser('~')+'/Dropbox/IsingData/'
            X, Y = load_data_set(root= root)
            print(X.shape)
            print(Y.shape)
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.8)
            print(x_train.shape)
            x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) # flatten each sample out
            x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
            x_train = minmax_scale(x_train) # this step is required in order to use cross-entropy loss for reconstruction
            x_test = minmax_scale(x_test) # scaling features in 0,1 interval
            train_loader = DataLoader(dataset=x_train, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
#            train_loader = infinite_dataset(train_loader)
            eval_loader = DataLoader(dataset=x_test, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
#            eval_loader = infinite_dataset(eval_loader)
            writer = SummaryWriter(os.path.join(cfg.model_dir, 'tensorboard'))
            model = BVAE(**cfg.BVAE).to(cfg.device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            for epoch in range(cfg.n_epochs):
                 print('= Starting epoch ', epoch, '/', cfg.n_epochs)
                 model, optimizer = train(model, optimizer, train_loader, cfg.batches_per_epoch, epoch, writer)
                 metrics = eval(eval_loader, model, cfg.batches_per_epoch * epoch, writer)
            save_metrics_params(metrics, cfg)
            torch.save(model.state_dict(), './trained_models/Ising_dense_%d'%())
