import torch
import numpy as np
from utilFuncs import to_np, to_torch
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from torch_sparse_solve import solve
from sys import exit

class Truss3DFE:
  def __init__(self, nodeXY, connectivity, bc):
    self.bc = bc
    self.nodeXY, self.numNodes = nodeXY, nodeXY.shape[0]
    self.ndof = 3*self.numNodes
    self.connectivity = connectivity
    self.numBars, self.numDOFPerBar = connectivity.shape[0], 6
    self.barCenter = 0.5*(nodeXY[connectivity[:,0]] + \
                          nodeXY[connectivity[:,1]])
    self.edofMat=np.zeros((self.numBars,self.numDOFPerBar),dtype=int);

    for br in range(self.numBars):
      n1, n2 = connectivity[br,0], connectivity[br,1]
      self.edofMat[br,:]=np.array([3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2])

    self.iK = np.kron(self.edofMat,np.ones((self.numDOFPerBar ,1))).flatten()
    self.jK = np.kron(self.edofMat,np.ones((1,self.numDOFPerBar))).flatten()
    bK = tuple(np.zeros((len(self.iK))).astype(int)) #batch values
    self.nodeIdx = [bK,self.iK.astype(int), self.jK.astype(int)]

    self.barLength = to_torch(np.sqrt((nodeXY[connectivity[:,0]][:,0] - \
                               nodeXY[connectivity[:,1]][:,0])**2 + \
                              (nodeXY[connectivity[:,0]][:,1] - \
                               nodeXY[connectivity[:,1]][:,1])**2 + \
                              (nodeXY[connectivity[:,0]][:,2] - \
                               nodeXY[connectivity[:,1]][:,2])**2 ))
    l = (nodeXY[connectivity[:,1]][:,0] - nodeXY[connectivity[:,0]][:,0])/ self.barLength
    m = (nodeXY[connectivity[:,1]][:,1] - nodeXY[connectivity[:,0]][:,1])/ self.barLength
    n = (nodeXY[connectivity[:,1]][:,2] - nodeXY[connectivity[:,0]][:,2])/ self.barLength
    L = np.zeros((2,6,self.numBars));
    L[0,0,:] = l;
    L[0,1,:] = m;
    L[0,2,:] = n;
    L[1,3,:] = l;
    L[1,4,:] = m;
    L[1,5,:] = n;

    self.L = to_torch(L);
    k0 = to_torch(np.array([[1.,-1.],[-1.,1.]]))

    self.K0 = torch.einsum('n,ijn->ijn', (1./self.barLength), torch.einsum('idn,ij,jfn->dfn',self.L,k0,self.L))
    self.K0 = torch.moveaxis(self.K0,-1,0)
    self.force = torch.zeros((self.ndof))
    self.applyForceOnNode(bc['forces'])
    self.fixNodes(bc['fixtures'])
    self.topFig, self.topAx = plt.subplots()
  #--------------------------#
  def fixNodes(self, fixed):
    self.fixedDofs = []
    self.fixedDofs = np.append(self.fixedDofs, 3*fixed['XNodes']).astype(int)
    self.fixedDofs = np.append(self.fixedDofs, 3*fixed['YNodes']+1).astype(int)
    self.fixedDofs = np.append(self.fixedDofs, 3*fixed['ZNodes']+2).astype(int)

    self.freeDofs = np.setdiff1d(np.arange(3*self.numNodes), self.fixedDofs)
    V = np.zeros((self.ndof, self.ndof))
    V[self.fixedDofs,self.fixedDofs] = 1.
    V = torch.tensor(V[np.newaxis])
    indices = torch.nonzero(V).t()
    values = V[indices[0], indices[1], indices[2]]
    penal = 1e9
    self.fixedBCPenaltyMatrix = \
        penal*torch.sparse_coo_tensor(indices, values, V.size())
  #--------------------------#
  def applyForceOnNode(self, force):
    self.force[3*force['nodes']] = force['fx']
    self.force[3*force['nodes']+1] = force['fy']
    self.force[3*force['nodes']+2] = force['fz']

  #--------------------------#
  def assembleK(self, E, A):
    sK = torch.einsum('i,ijk->ijk',E*A, self.K0)
    sK = sK.flatten()
    Kasm = torch.sparse_coo_tensor(np.array(self.nodeIdx), sK,\
                            (1, self.ndof, self.ndof))
    return Kasm
  #--------------------------#
  def solveFE(self, E, A):
    self.E, self.A = E, A
    f = self.force
    f = self.force.double().unsqueeze(0).unsqueeze(2)
    Kasm = self.assembleK(E, A);

    K = (Kasm + np.max(to_np(E))*self.fixedBCPenaltyMatrix).coalesce()

    solveType = 'Dense'

    if solveType == 'Dense':
        KDense = K.to_dense()
        u1 = torch.linalg.solve(KDense,f[0,:])
        self.u = u1[0,:,0]
    else:
        self.u = solve(K, f).flatten()


    self.dispX, self.dispY, self.dispZ = self.u[0::3], self.u[1::3], self.u[2::3]


    q1 = torch.cat((self.dispX[self.connectivity[:,1]],self.dispY[self.connectivity[:,1]],self.dispZ[self.connectivity[:,1]]),0).reshape((3,-1))
    q0 = torch.cat((self.dispX[self.connectivity[:,0]],self.dispY[self.connectivity[:,0]],self.dispZ[self.connectivity[:,0]]),0).reshape((3,-1))
    q = torch.cat((q0,q1)).float()

    self.nodalDeformation = torch.sqrt(self.dispX**2 + self.dispY**2 + self.dispZ**2)
    temp = torch.einsum('ijk,jk->ik',self.L,q)

    stress = torch.einsum('j,jk->k',torch.tensor([-1.,1]),temp)
    self.internalForce = stress*E*A/self.barLength;
    return self.u, self.dispX, self.dispY, self.dispZ, self.nodalDeformation, self.internalForce
  #--------------------------#
  def computeCompliance(self, u):
    J = torch.einsum('i,i->i',u, self.force).sum()
    return J
  #--------------------------#
  def plot(self, titleStr = 'None', plotDeformed = True):
    def set_axes_equal(ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    plt.ion()
    plt.clf()
    ax = plt.axes(projection='3d')
    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.set_aspect('auto')
    # set_axes_equal(ax)
    ax.set_axis_off()
    A = to_np(self.A)
    Amin = np.min(A);
    LScale = torch.max(self.barLength);

    scale = 0.15*LScale/np.max(to_np(self.nodalDeformation))

    # plot the bars undeformed and if true then deformed
    for i in range(self.numBars):
      n1, n2 = self.connectivity[i,0], self.connectivity[i,1]
      sx, sy, sz = self.nodeXY[n1][0], self.nodeXY[n1][1], self.nodeXY[n1][2]
      ex, ey, ez = self.nodeXY[n2][0], self.nodeXY[n2][1], self.nodeXY[n2][2]
      sx, sy, sz = self.nodeXY[n1,0], self.nodeXY[n1,1], self.nodeXY[n1,2]
      ex, ey, ez = self.nodeXY[n2,0], self.nodeXY[n2,1], self.nodeXY[n2,2]


      thkns = 1 + 3.5*np.log(A[i]/Amin); #10 for prob 1
      t_or_c = 't';#*(self.internalForce[i] > 0.) + 'c'*(self.internalForce[i] <= 0.)
      if(plotDeformed == False):
          clr = 'blue' #'blue'*(self.internalForce[i] <= 0.) + 'red'*(self.internalForce[i] > 0.)
          ax.plot3D([sx,ex],[sy,ey],[sz,ez], color = 'black', linewidth = thkns, alpha = 1)
          ax.set_axis_off()
          ax.azim = 90
          ax.dist = 10
          ax.elev = 90
          # plt.text(0.5*(sx+ex), 0.5*(sy+ey), '$A_{:d}$'.format(i), \
          #           rotation=180.*self.barOrientation[i]/np.pi, size = 24)
      else:
          clr = 'blue'*(self.internalForce[i] <= 0.) + 'red'*(self.internalForce[i] > 0.)
          ax.plot3D([sx,ex],[sy,ey], [sz,ez], color = 'black', linewidth = thkns, alpha = 1)
          if(plotDeformed):
            dx1, dx2 = self.u[3*n1], self.u[3*n2]
            dy1, dy2 = self.u[3*n1+1], self.u[3*n2+1]
            dz1, dz2 = self.u[3*n1+2], self.u[3*n2+2]

            ax.plot3D([sx + scale*dx1,ex + scale*dx2],\
                     [sy + scale*dy1,ey + scale*dy2], \
                     [sz + scale*dz1,ez + scale*dz2], \
                     color = 'black', linestyle = 'dashed')
            ax.azim = -90
            ax.dist = 10
            ax.elev = 90
          # plt.text(0.5*(sx+ex), 0.5*(sy+ey), '$A_{:d}^{:s}$ = {:.1e}'.format(i,t_or_c, self.A[i]), \
          #          rotation=180.*self.barOrientation[i]/np.pi)
      # make the panes transparent

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
    # plt.axis('Equal')
    plt.title(titleStr)
    plt.grid(False)
    plt.pause(0.01)
  #--------------------------#
  def getVolume(self, A):
    return torch.einsum('i,i->i',to_torch(self.barLength), A).sum()
  #--------------------------#
