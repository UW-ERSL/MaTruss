import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import time
#--------------------------#
def set_seed(manualSeed):
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.manual_seed(manualSeed)
  torch.cuda.manual_seed(manualSeed)
  torch.cuda.manual_seed_all(manualSeed)
  np.random.seed(manualSeed)
  random.seed(manualSeed)
#--------------------------#
def to_np(x):
  return x.detach().cpu().numpy()
#--------------------------#
def to_torch(x):
  return torch.tensor(x).float()
#--------------------------#
def plotConvergence(convg):
  for key in convg:
    plt.figure()
    y = np.array(convg[key])
    plt.plot(y[1:], label = str(key))
    plt.xlabel('Iterations')
    plt.ylabel(str(key))
    plt.grid('True')
    
#--------------------------#
def plotConstraintsConvergence(convg, saveFileName):
    strokes = ['--', '-.', '-', ':']
    labelStr = ['buckling constraint', 'yielding constraint', 'cost constraint']
    plt.figure()
    for ctr, key in enumerate(convg):
        y = np.array(convg[key])
        plt.plot(y[1:], strokes[ctr], label = labelStr[ctr])
        plt.xlabel('Iterations')
        plt.ylabel('Value')
        plt.grid('True')
    plt.legend()
    plt.title('Convergence of constraints')
    plt.savefig(saveFileName)
    plt.show()
#--------------------------#
def plotObjectiveConvergence(convg, saveFileName):
    strokes = ['-', '.', '-', ':']
    plt.figure()
    for ctr, key in enumerate(convg):
        y = np.array(convg[key])
        plt.plot(y[1:], strokes[ctr], label = str(key))
        plt.xlabel('Iterations')
        plt.ylabel('Value')
        plt.grid('True')
    plt.legend()
    plt.title('Convergence of objective')

    plt.savefig(saveFileName)
    plt.show()

#--------------------------#
def timing(f):
  def wrap(*args, **kwargs):
    time1 = time.time()
    ret = f(*args, **kwargs)
    time2 = time.time()
    print('{:s} function took {:.3f} sec'.format(f.__name__, (time2-time1)))

    return ret
  return wrap