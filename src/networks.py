import torch
import torch.nn as nn
import torch.nn.functional as F
from utilFuncs import set_seed

#%%
class Encoder(nn.Module):
  def __init__(self, encoderSettings):
    super(Encoder, self).__init__()
    set_seed(1234)
    self.linear1 = nn.Linear(encoderSettings['inputDim'], encoderSettings['hiddenDim'])
    self.linear2 = nn.Linear(encoderSettings['hiddenDim'], encoderSettings['latentDim'])
    self.linear3 = nn.Linear(encoderSettings['hiddenDim'], encoderSettings['latentDim'])
    
    self.N = torch.distributions.Normal(0, 1)
    self.kl = 0
    self.isTraining = False
  def forward(self, x):
    x = F.relu(self.linear1(x))
    mu =  self.linear2(x)
    sigma = torch.exp(self.linear3(x))
    if(self.isTraining):
      self.z = mu + sigma*self.N.sample(mu.shape)
    else:
      self.z = mu

    self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
    return self.z
#--------------------------#
class Decoder(nn.Module):
  def __init__(self, decoderSettings):
    super(Decoder, self).__init__()
    self.linear1 = nn.Linear(decoderSettings['latentDim'], decoderSettings['hiddenDim'])
    self.linear2 = nn.Linear(decoderSettings['hiddenDim'], decoderSettings['outputDim'])

  def forward(self, z):
    z = F.relu(self.linear1(z)) # 
    z = torch.sigmoid(self.linear2(z)) # decoder op in range [0,1]
    return z
#--------------------------#
class VariationalAutoencoder(nn.Module):
  def __init__(self, vaeSettings):
    super(VariationalAutoencoder, self).__init__()
    self.encoder = Encoder(vaeSettings['encoder'])
    self.decoder = Decoder(vaeSettings['decoder'])

  def forward(self, x):
    z = self.encoder(x)
    return self.decoder(z)
#--------------------------#
#%%
class MaterialNetwork(nn.Module):
  def __init__(self, nnSettings):
    self.nnSettings = nnSettings
    super().__init__()
    self.layers = nn.ModuleList()
    set_seed(1234)
    current_dim = nnSettings['inputDim']
    for lyr in range(nnSettings['numLayers']): # define the layers
      l = nn.Linear(current_dim, nnSettings['numNeuronsPerLyr'])
      nn.init.xavier_normal_(l.weight)
      nn.init.zeros_(l.bias)
      self.layers.append(l)
      current_dim = nnSettings['numNeuronsPerLyr']
    self.layers.append(nn.Linear(current_dim, nnSettings['outputDim']))
    self.bnLayer = nn.ModuleList()
    for lyr in range(nnSettings['numLayers']): # batch norm
      self.bnLayer.append(nn.BatchNorm1d(nnSettings['numNeuronsPerLyr']))

  def forward(self, x):
    m = nn.LeakyReLU();
    ctr = 0;
    for layer in self.layers[:-1]: # forward prop
      x = m(layer(x))#m(self.bnLayer[ctr](layer(x)));
      ctr += 1;
    opLayer = self.layers[-1](x)
    nnOut = torch.sigmoid(opLayer)
    z = self.nnSettings['zMin'] + self.nnSettings['zRange']*nnOut 
    return z

#--------------------------#
#%%
class TopologyNetwork(nn.Module):
  def __init__(self, nnSettings):
    self.inputDim = nnSettings['inputDim']# x and y coordn of the point
    self.outputDim = nnSettings['outputDim']
    super().__init__()
    self.layers = nn.ModuleList()
    set_seed(1234)
    current_dim = self.inputDim
    for lyr in range(nnSettings['numLayers']): # define the layers
      l = nn.Linear(current_dim, nnSettings['numNeuronsPerLyr'])
      nn.init.xavier_normal_(l.weight)
      nn.init.zeros_(l.bias)
      self.layers.append(l)
      current_dim = nnSettings['numNeuronsPerLyr']
    self.layers.append(nn.Linear(current_dim, self.outputDim))
    self.bnLayer = nn.ModuleList()
    for lyr in range(nnSettings['numLayers']): # batch norm
      self.bnLayer.append(nn.BatchNorm1d(nnSettings['numNeuronsPerLyr']))

  def forward(self, x):
    m = nn.LeakyReLU()
    ctr = 0
    for layer in self.layers[:-1]: # forward prop
      x = m(self.bnLayer[ctr](layer(x)))
      ctr += 1
    opLayer = self.layers[-1](x)
    rho = torch.sigmoid(opLayer).view(-1)
    return rho