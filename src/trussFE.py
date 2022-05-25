import torch
import numpy as np
from utilFuncs import to_np, to_torch
import matplotlib.pyplot as plt
from torch_sparse_solve import solve

class TrussFE:
  def __init__(self, nodeXY, connectivity, bc):
    self.bc = bc
    self.nodeXY, self.numNodes = nodeXY, nodeXY.shape[0]
    self.ndof = 2*self.numNodes
    self.connectivity = connectivity
    self.numBars, self.numDOFPerBar = connectivity.shape[0], 4
    self.barCenter = 0.5*(nodeXY[connectivity[:,0]] + \
                          nodeXY[connectivity[:,1]])
    self.edofMat=np.zeros((self.numBars,self.numDOFPerBar),dtype=int);

    for br in range(self.numBars):
      n1, n2 = connectivity[br,0], connectivity[br,1]
      self.edofMat[br,:]=np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1])

    self.iK = np.kron(self.edofMat,np.ones((self.numDOFPerBar ,1))).flatten()
    self.jK = np.kron(self.edofMat,np.ones((1,self.numDOFPerBar))).flatten()
    bK = tuple(np.zeros((len(self.iK))).astype(int)) #batch values
    self.nodeIdx = [bK,self.iK.astype(int), self.jK.astype(int)]

    self.barLength = to_torch(np.sqrt( (nodeXY[connectivity[:,0]][:,0] - \
                               nodeXY[connectivity[:,1]][:,0])**2 + \
                              (nodeXY[connectivity[:,0]][:,1] - \
                               nodeXY[connectivity[:,1]][:,1])**2 ))

    self.barOrientation = to_torch(np.arctan2((nodeXY[connectivity[:,1]][:,1] - \
                                  nodeXY[connectivity[:,0]][:,1]),\
                                     (nodeXY[connectivity[:,1]][:,0] - \
                                  nodeXY[connectivity[:,0]][:,0])))

    R = np.einsum('i,jk->ijk',np.cos(self.barOrientation),\
                              np.array([[1,0,0,0],[0,0,1,0]])) +\
        np.einsum('i,jk->ijk',np.sin(self.barOrientation),\
                              np.array([[0,1,0,0],[0,0,0,1]]))

    k0 = np.array([[1.,-1.],[-1.,1.]])

    self.K0 = torch.tensor(np.einsum('b,bfr->bfr', 1./self.barLength, \
                        np.einsum('bfw, bwr->bfr',\
                        np.einsum('bft, tw->bfw', R.swapaxes(1,2), k0), R)))
    self.force = torch.zeros((self.ndof))
    self.applyForceOnNode(bc['forces'])
    self.fixNodes(bc['fixtures'])
    self.topFig, self.topAx = plt.subplots()
  #--------------------------#
  def fixNodes(self, fixed):
    self.fixedDofs = []
    self.fixedDofs = np.append(self.fixedDofs, 2*fixed['XNodes']).astype(int)
    self.fixedDofs = np.append(self.fixedDofs, 2*fixed['YNodes']+1).astype(int)
    self.freeDofs = np.setdiff1d(np.arange(2*self.numNodes), self.fixedDofs)
    V = np.zeros((self.ndof, self.ndof))
    V[self.fixedDofs,self.fixedDofs] = 1.
    V = torch.tensor(V[np.newaxis])
    indices = torch.nonzero(V).t()
    values = V[indices[0], indices[1], indices[2]]
    penal = 1e5
    self.fixedBCPenaltyMatrix = \
        penal*torch.sparse_coo_tensor(indices, values, V.size())
  #--------------------------#
  def applyForceOnNode(self, force):
    self.force[2*force['nodes']] = force['fx']
    self.force[2*force['nodes']+1] = force['fy']
  #--------------------------#
  def assembleK(self, E, A):
    sK = torch.einsum('i,ijk->ijk',E*A, self.K0)
    # print(sK)
    sK = sK.flatten()
    Kasm = torch.sparse_coo_tensor(self.nodeIdx, sK,\
                            (1, self.ndof, self.ndof))
    return Kasm
  #--------------------------#
  def solveFE(self, E, A):
    self.E, self.A = E, A
    f = self.force.double().unsqueeze(0).unsqueeze(2)
    Kasm = self.assembleK(E, A);
    K = (Kasm + np.max(to_np(E))*self.fixedBCPenaltyMatrix).coalesce()
    self.u = solve(K, f).flatten()

    self.dispX, self.dispY = self.u[0::2], self.u[1::2]
    self.nodalDeformation = torch.sqrt(self.dispX**2 + self.dispY**2)
    du = (self.dispX[self.connectivity[:,1]] - self.dispX[self.connectivity[:,0]])*torch.cos(self.barOrientation)
    dv = (self.dispY[self.connectivity[:,1]] - self.dispY[self.connectivity[:,0]])*torch.sin(self.barOrientation)
    self.internalForce = (E*A*(du+dv))/(self.barLength)
    return self.u, self.dispX, self.dispY, self.nodalDeformation, self.internalForce
  #--------------------------#
  def computeCompliance(self, u):
    J = torch.einsum('i,i->i',u, self.force).sum()
    return J
  #--------------------------#
  def plot(self, titleStr, plotDeformed = True):
    plt.ion()
    plt.clf()
    A = to_np(self.A)
    Amin = np.min(A);
    LScale = torch.max(self.barLength);

    scale = 0.15*LScale/np.max(to_np(self.nodalDeformation))

    # plot the bars undeformed and if true the deformed
    for i in range(self.numBars):
      n1, n2 = self.connectivity[i,0], self.connectivity[i,1]
      sx, sy = self.nodeXY[n1][0], self.nodeXY[n1][1]
      ex, ey = self.nodeXY[n2][0], self.nodeXY[n2][1]

      thkns = 1 + 10.*np.log(A[i]/Amin);
      t_or_c = 't'*(self.internalForce[i] > 0.) + 'c'*(self.internalForce[i] <= 0.)
      if(plotDeformed == False):
          clr = 'blue'*(self.internalForce[i] <= 0.) + 'red'*(self.internalForce[i] > 0.)
          plt.plot([sx,ex],[sy,ey], color = 'black', linewidth = thkns, alpha = 0.5)
          # plt.text(0.5*(sx+ex), 0.5*(sy+ey), '$A_{:d}$'.format(i), \
          #           rotation=180.*self.barOrientation[i]/np.pi, size = 24)
      else:
          clr = 'blue'*(self.internalForce[i] <= 0.) + 'red'*(self.internalForce[i] > 0.)
          plt.plot([sx,ex],[sy,ey], color = 'black', linewidth = thkns, alpha = 0.5)
          if(plotDeformed):
            dx1, dx2 = self.u[2*n1], self.u[2*n2]
            dy1, dy2 = self.u[2*n1+1], self.u[2*n2+1]
            plt.plot([sx + scale*dx1,ex + scale*dx2],\
                     [sy + scale*dy1,ey + scale*dy2], \
                     color = 'black', linestyle = 'dashed',)
          # plt.text(0.5*(sx+ex), 0.5*(sy+ey), '$A_{:d}^{:s}$ = {:.1e}'.format(i,t_or_c, self.A[i]), \
          #          rotation=180.*self.barOrientation[i]/np.pi)

    # number the nodes
    annotateNodes = False
    if(annotateNodes):
      for i in range(self.numNodes):
        plt.annotate('{:d}'.format(i), (self.nodeXY[i,0], self.nodeXY[i,1]))

    showFixtures = False
    if(showFixtures):
    # show fixtures
      for i in self.bc['fixtures']['XNodes']:
        plt.scatter(self.nodeXY[i,0], self.nodeXY[i,1], marker=4 ,s=100, c = 'orange')
      for i in self.bc['fixtures']['YNodes']:
        plt.scatter(self.nodeXY[i,0], self.nodeXY[i,1], marker=6,s=100, c = 'green')

    # show forces
    showForces = False
    if(showForces):
      for ctr, nd in enumerate(self.bc['forces']['nodes']):
        plt.quiver(self.nodeXY[nd,0], self.nodeXY[nd,1], \
                    self.bc['forces']['fx'][ctr],self.bc['forces']['fy'][ctr], color = 'purple')

    self.topFig.canvas.draw()
    plt.axis('Equal')
    plt.title(titleStr)
    plt.grid(False)
    plt.pause(0.01)
  #--------------------------#
  def getVolume(self, A):
    return torch.einsum('i,i->i',self.barLength, A).sum()
  #--------------------------#
