from networks import VariationalAutoencoder
import torch
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
import numpy as np
from utilFuncs import to_np
import pickle

#--------------------------#
class MaterialEncoder:
  def __init__(self, trainingData, dataInfo, dataIdentifier, vaeSettings):
    self.trainingData, self.dataInfo = trainingData, dataInfo
    self.dataIdentifier = dataIdentifier
    self.vaeSettings = vaeSettings
    self.vaeNet = VariationalAutoencoder(vaeSettings)
  #--------------------------#
  def loadAutoencoderFromFile(self, fileName):
    with open('./results/vaeTrained.pkl', 'r') as f:
      obj0 = pickle.load(f)
      self.vaeNet.load_state_dict(torch.load(obj0))
      self.vaeNet.encoder.isTraining = False
  #--------------------------#
  def trainAutoencoder(self, numEpochs, klFactor, savedNet, learningRate):
    opt = torch.optim.Adam(self.vaeNet.parameters(), learningRate)
    convgHistory = {'reconLoss':[], 'klLoss':[], 'loss':[]}
    self.vaeNet.encoder.isTraining = True
    for epoch in range(numEpochs):
      opt.zero_grad()
      predData = self.vaeNet(self.trainingData)
      klLoss = klFactor*self.vaeNet.encoder.kl

      reconLoss =  ((self.trainingData - predData)**2).sum()
      loss = reconLoss + klLoss 
      loss.backward()
      convgHistory['reconLoss'].append(reconLoss)
      convgHistory['klLoss'].append(klLoss/klFactor) # save unscaled loss
      convgHistory['loss'].append(loss)
      opt.step()
      if(epoch%500 == 0):
        print('Iter {:d} reconLoss {:.2E} klLoss {:.2E} loss {:.2E}'.\
              format(epoch, reconLoss.item(), klLoss.item(), loss.item()))
    self.vaeNet.encoder.isTraining = False
    with open('./results/vaeTrained.pkl', 'wb+') as f:
      pickle.dump([self.vaeNet.encoder.state_dict()], f)
    return convgHistory
  #--------------------------#
  def getClosestMaterialFromZ(self, z, numClosest = 1):
    zData = self.vaeNet.encoder.z.to('cpu').detach().numpy()
    dist = np.linalg.norm(zData- to_np(z), axis = 1)
    meanDist = np.max(dist)
    distOrder = np.argsort(dist)
    matToUseFromDB = {'material':[], 'confidence':[]}
    for i in range(numClosest):
      mat = self.dataIdentifier['name'][distOrder[i]]
      matToUseFromDB['material'].append(mat)
      confidence = 100.*(1.- (dist[distOrder[i]]/meanDist))
      matToUseFromDB['confidence'].append(confidence)
      print(f"closest material {i} : {mat} , confidence {confidence:.2F}")
    return matToUseFromDB
  #--------------------------#
  def plotLatent(self, ltnt1, ltnt2, plotHull, annotateHead, saveFileName):
    clrs = ['purple', 'green', 'orange', 'pink', 'yellow', 'black', 'violet', 'cyan', 'red', 'blue']
    colorcol = self.dataIdentifier['classID']
    ptLabel = self.dataIdentifier['name']
    autoencoder = self.vaeNet
    z = autoencoder.encoder.z.to('cpu').detach().numpy()
    fig, ax = plt.subplots()

    for i in range(np.max(colorcol)+1): 
      zMat = np.vstack((z[colorcol == i,ltnt1], z[colorcol == i,ltnt2])).T
      ax.scatter(zMat[:, 0], zMat[:, 1], c = 'black', s = 4)#clrs[i]
      if(i == np.max(colorcol)): #removed for last class TEST
        break # END TEST
      if(plotHull):
        hull = ConvexHull(zMat)
        cent = np.mean(zMat, 0)
        pts = []
        for pt in zMat[hull.simplices]:
            pts.append(pt[0].tolist())
            pts.append(pt[1].tolist())
  
        pts.sort(key=lambda p: np.arctan2(p[1] - cent[1],
                                        p[0] - cent[0]))
        pts = pts[0::2]  # Deleting duplicates
        pts.insert(len(pts), pts[0])
        poly = Polygon(1.1*(np.array(pts)- cent) + cent,
                       facecolor= clrs[i], alpha=0.1, edgecolor = 'black') #'black'
        poly.set_capstyle('round')
        plt.gca().add_patch(poly)
        ax.annotate(self.dataIdentifier['className'][i], (cent[0], cent[1]), size = 12)
    for i, txt in enumerate(ptLabel):
      if(annotateHead == False or ( annotateHead == True and  i<np.max(colorcol)+1)):
        
        # continue
        ax.annotate(txt, (z[i,ltnt1], z[i,ltnt2]), size = 6)

  #   plt.axis('off')
    # ticks = [-2.5, -2, -1.5, -1., -0.5, 0., 0.5, 1., 1.5]
    # ticklabels = ['-2.5','-2','-1.5', '-1', '-0.5', '0','0.5', '1', '1.5']
    # plt.xticks(ticks, ticklabels, fontsize=18)
    # plt.yticks(ticks, ticklabels, fontsize=18)
    plt.xlabel('z{:d}'.format(ltnt1), size = 18)
    plt.ylabel('z{:d}'.format(ltnt2), size = 18)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(saveFileName)
    
    return fig, ax
  
    
