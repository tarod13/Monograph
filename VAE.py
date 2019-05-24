# Source: https://ml-cheatsheet.readthedocs.io/en/latest/architectures.html

from __future__ import print_function
import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from sys import stdout
import time


def where(cond, x_1, x_2):
    cond = cond.float()    
    return (cond * x_1) + ((1-cond) * x_2)

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('linear') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.01)

# Loss function
# def vae_loss(output, input, mean, logvar, loss_func=[], mode='continuous'):
#     if mode == 'continuous':
#         recon_loss = gaussian_MSELoss(output, input)
#     else:
#         recon_loss = loss_func(output, input)
#     kl_loss = torch.mean(0.5 * torch.sum(
#         torch.exp(logvar) + mean**2 - 1. - logvar, 1))
#     return 2*recon_loss + 0.5*kl_loss

# Loss function
def vae_loss(output, input, mean, logvar, loss_func, alpha=0.25):
    recon_loss = loss_func(output, input)
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mean**2 - 1. - logvar, 1))
    return 2*recon_loss + alpha*kl_loss

def gaussian_MSELoss(output, input):
    mean, stdev = output
    return 0.5*(((input-mean)/stdev)**2).mean(0).sum()

# VAE network
class ConvVAE(nn.Module):
    def __init__(self, channel_dim, latent_dim, input_dim, mode='continuous', min_log_std=-20, max_log_std=2):
        super().__init__()
        self.C = channel_dim
        self.latent_dim = latent_dim
        self.mode = mode
        if self.mode == 'discrete':
            self.loss_func = nn.BCELoss()        
        self.batch_size = 100
        self.num_epochs = 100
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.verbose = True

        # Encoder layers
        self.conv1 =  nn.Conv2d(self.C, 16, kernel_size=3, stride=3, padding=1)         # N x  16 x 22 x 22        
        self.conv2 =  nn.Conv2d(16, 128, kernel_size=3, stride=3, padding=1)            # N x 128 x  8 x  8
        
        # Latent variable layers
        self.z_mean = nn.Linear(8192, latent_dim) # 4608 for size 32
        self.z_log_var = nn.Linear(8192, latent_dim)
        self.z_develop = nn.Linear(latent_dim, 8192)

        # Decoder layers
        self.convT2 = nn.ConvTranspose2d(128, 16, kernel_size=3, stride=3, padding=1)           # N x 16 x 22 x 22                                    
        self.convT1_mean = nn.ConvTranspose2d(16, self.C, kernel_size=3, stride=3, padding=1)   # N x  2 x 64 x 64
        self.convT1_mean.weight.data.uniform_(-3e-3, 3e-3)
        
        if self.mode == 'continuous':
            self.convT1_logstdev = nn.ConvTranspose2d(16, self.C, kernel_size=3, stride=3, padding=1)    # N x  2 x 64 x 64
            self.convT1_logstdev.weight.data.uniform_(-3e-3, 3e-3)            
                             
        
        #self.apply(weights_init)
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr = 0.002, momentum=0.9)

    def encoder(self, x):        
        x = F.leaky_relu(self.conv1(x))       
        x = F.leaky_relu(self.conv2(x))     
        return x
    
    def decoder(self, x):
        x = F.leaky_relu(self.convT2(x)) 
        m = self.convT1_mean(x)
        if self.mode == 'continuous':
            log_stdev = self.convT1_logstdev(x)
            log_stdev = torch.clamp(log_stdev, -20,20)
            stdev = torch.exp(log_stdev)
            return [m,stdev]
        else:
            x = self.dropout2(torch.sigmoid(m))     
            return x

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mean = self.z_mean(x)
        log_var = self.z_log_var(x)
        return mean, log_var

    def sample_z(self, mean, logvar):
        stddev = torch.exp(0.5 * logvar)
        noise = torch.randn(stddev.size()).to(self.device)
        return (noise * stddev) + mean

    def decode(self, z):
        out = F.leaky_relu(self.z_develop(z))        
        out = out.view(z.size(0), 128, 8, -1)
        out = self.decoder(out)
        return out

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        out = self.decode(z)
        return out, mean, logvar
    
    def forward_z(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        return(z, mean, logvar)

    def forward_x(self, z):
        x = self.decode(z).cpu()
        return(x)
    
    def train(self, dataset, alpha=0.25):  
        losses = []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, num_workers=1)  
        
        # For each epoch
        for epoch in range(0, self.num_epochs):
            # For each batch in the dataloader
            for i, batch in enumerate(dataloader, 0):

                self.zero_grad()
                batch = batch.float().to(self.device)                               
                outputs, means, logvars = self(batch)
                if self.mode == 'discrete':
                    loss = vae_loss(outputs, batch, means, logvars, self.loss_func, self.mode)
                else:
                    loss = vae_loss(outputs, batch, means, logvars)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

                # Output training stats
                if i % 200 == 0 and epoch % 20 == 0 and self.verbose:
                    print('[%d/%d][%d/%d]\tLoss: %.6f' % (epoch, self.num_epochs, i, len(dataloader), loss.item()))                  
        return(losses)


class MLPVAE(nn.Module):
    def __init__(self, n_channels, input_dim, latent_dim, mode='continuous', min_log_std=-20, max_log_std=2):
        super().__init__()  
        self.n_channels = n_channels     
        self.input_dim = input_dim 
        self.latent_dim = latent_dim
        self.mode = mode
        if self.mode == 'discrete':
            self.loss_func = nn.BCELoss()
        else:
            self.loss_func = nn.MSELoss()        
        self.batch_size = 50
        self.num_epochs = 100
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.verbose = True

        # Encoder layers
        self.encoder_hidden =  nn.Linear(n_channels*input_dim*input_dim, 256)
        self.l1 =  nn.Linear(256, 64)            
        
        # Latent variable layers
        self.z_mean = nn.Linear(64, latent_dim)
        self.z_log_var = nn.Linear(64, latent_dim)
        self.z_develop = nn.Linear(latent_dim, 64)
        self.z_mean.weight.data.uniform_(-3e-3, 3e-3)
        self.z_mean.bias.data.uniform_(-3e-3, 3e-3)
        self.z_log_var.weight.data.uniform_(-3e-3, 3e-3)
        self.z_log_var.bias.data.uniform_(-3e-3, 3e-3)

        # Decoder layers
        self.l2 =  nn.Linear(64, 256)  
        self.decoder_hidden =  nn.Linear(256, n_channels*input_dim*input_dim)
        self.decoder_hidden.weight.data.uniform_(-3e-3, 3e-3)
        self.decoder_hidden.bias.data.uniform_(-3e-3, 3e-3)
        # self.decoder_hidden_logstdev =  nn.Linear(256, n_channels*input_dim*input_dim)        
        # self.decoder_hidden_logstdev.weight.data.uniform_(-3e-3, 3e-3)
        # self.decoder_hidden_logstdev.bias.data.uniform_(-3e-3, 3e-3)
        
        # self.apply(weights_init)
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.to(self.device)
        if self.mode == 'discrete':
            # self.optimizer = optim.SGD(self.parameters(), lr = 0.005, momentum=0.9)
            self.optimizer = optim.Adam(self.parameters(), lr = 0.005)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr = 0.00005, momentum=0.9)


    def encoder(self, x):        
        x = F.leaky_relu(self.encoder_hidden(x))
        x = F.leaky_relu(self.l1(x))
        return x
    
    def decoder(self, x):
        x = F.leaky_relu(self.l2(x))
        if self.mode == 'continuous':
            x = self.decoder_hidden(x)
            # log_stdev = self.decoder_hidden_logstdev(x)
            # stdev = torch.exp(torch.clamp(log_stdev, -20,20))
            # return [m,stdev]
        else:
            x = torch.sigmoid(self.decoder_hidden(x))        
        return x

    def encode(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)        
        mean = self.z_mean(x)
        log_var = torch.clamp(self.z_log_var(x), self.min_log_std, self.max_log_std)
        return mean, log_var

    def sample_z(self, mean, logvar):
        stddev = torch.exp(0.5 * logvar)
        noise = torch.randn(stddev.size()).to(self.device)
        return (noise * stddev) + mean

    def decode(self, z):
        out = F.leaky_relu(self.z_develop(z))        
        out = self.decoder(out)
        # if self.mode == 'continuous':
        #     m, s = out
        #     m = m.view(z.size(0), self.n_channels, self.input_dim, -1)
        #     s = s.view(z.size(0), self.n_channels, self.input_dim, -1)
        #     out = [m,s] 
        # else:
        out = out.view(z.size(0), self.n_channels, self.input_dim, -1)
        return out

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        out = self.decode(z)
        return out, mean, logvar
    
    def forward_z(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        return(z, mean, logvar)

    def forward_x(self, z):
        x = self.decode(z).cpu()
        return(x)
    
    def train(self, dataset, alpha=0.25):  
        losses = []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, num_workers=1)  
        
        # For each epoch
        for epoch in range(0, self.num_epochs):
            # For each batch in the dataloader
            for i, batch in enumerate(dataloader, 0):

                self.zero_grad()
                batch = batch.float().to(self.device)                               
                outputs, means, logvars = self(batch)
                # if self.mode == 'discrete':
                #     loss = vae_loss(outputs, batch, means, logvars, self.loss_func, self.mode)
                # else:
                #     loss = vae_loss(outputs, batch, means, logvars)
                loss = vae_loss(outputs, batch, means, logvars, self.loss_func, alpha=alpha)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

                # Output training stats
                if i % 200 == 0 and epoch % 20 == 0 and self.verbose:
                    print('[%d/%d][%d/%d]\tLoss: %.6f' % (epoch, self.num_epochs, i, len(dataloader), loss.item()))                  
        return(losses)


class MLPAE(nn.Module):
    def __init__(self, n_channels, input_dim, latent_dim, mode='continuous'):
        super().__init__()  
        self.n_channels = n_channels    
        self.input_dim = input_dim  
        self.latent_dim = latent_dim
        self.mode = mode
        if mode == 'continuous':
            self.loss_func = nn.MSELoss()
        else:
            self.loss_func = nn.BCELoss()        
        self.batch_size = 50
        self.num_epochs = 100
        self.verbose = True

        # Encoder layers
        self.encoder_hidden =  nn.Linear(n_channels*input_dim*input_dim, 256)
        self.l1 = nn.Linear(256, 64)      

        # Latent variable layers
        self.z_mean = nn.Linear(64, latent_dim)
        self.z_develop = nn.Linear(latent_dim, 64)

        # Decoder layers
        self.l2 = nn.Linear(64, 256)
        self.decoder_hidden =  nn.Linear(256, n_channels*input_dim*input_dim)

        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.to(self.device)
        self.optimizer = optim.SGD(self.parameters(), lr = 0.00005, momentum=0.9)

    def encoder(self, x):        
        x = F.leaky_relu(self.encoder_hidden(x))
        x = F.leaky_relu(self.l1(x))
        return x
    
    def decoder(self, x):
        x = F.leaky_relu(self.l2(x))
        x = self.decoder_hidden(x)
        if self.mode == 'discrete':
            x = torch.sigmoid(x)                
        return x

    def encode(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)        
        z = self.z_mean(x)
        return z

    def decode(self, z):
        out = F.leaky_relu(self.z_develop(z))        
        out = self.decoder(out)
        out = out.view(z.size(0), self.n_channels, self.input_dim, -1)
        return out

    def forward(self, x):
        z = self.encode(x)        
        out = self.decode(z)
        return out
    
    def forward_z(self, x):
        z = self.encode(x)        
        return(z)

    def forward_x(self, z):
        x = self.decode(z).cpu()
        return(x)
    
    def train(self, dataset):  
        losses = []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, num_workers=1)  
        
        # For each epoch
        for epoch in range(0, self.num_epochs):
            # For each batch in the dataloader
            for i, batch in enumerate(dataloader, 0):

                self.zero_grad()
                batch = batch.float().to(self.device)                               
                outputs = self(batch)
                loss = self.loss_func(outputs, batch)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

                # Output training stats
                if i % 200 == 0 and epoch % 20 == 0 and self.verbose:
                    print('[%d/%d][%d/%d]\tLoss: %.6f' % (epoch, self.num_epochs, i, len(dataloader), loss.item()))                  
        return(losses)
