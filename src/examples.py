import numpy as np
import torch
def getExample(exampleNo):
    if(exampleNo == 1):
        exampleName = 'bridge'
        nodeXY = np.array([0.5, 1., 1.5, 1., 1., 0., 0., 0., 2., 0.]).reshape(-1,2)
        connectivity = np.array([0, 1, 0, 2, 0, 3, 1, 2, 1, 4, 2, 3, 2, 4]).reshape(-1,2)
        fixed = {'XNodes':np.array([3,4]), 'YNodes':np.array([3,4])}
        force = {'nodes':np.array([0,1]), 'fx':1.E3*torch.tensor([1.,2.]), 'fy':1.E3*torch.tensor([-2.,0.])}

    if(exampleNo == 2):
        exampleName = 'twoBarTriangle'
        nodeXY = np.array([0.,0.,0.5,0.,0.25,0.5]).reshape(-1,2)
        connectivity = np.array([0,2,1,2]).reshape(-1,2)
        fixed = {'XNodes':np.array([0,1]), 'YNodes':np.array([0,1])}
        force = {'nodes':np.array([2]), 'fx':1.E3*torch.tensor([500.]), 'fy':1.E3*torch.tensor([125.])}

    if(exampleNo == 3):
        exampleName = 'paramterizedTwoBar'
        theta, beta = np.pi/3., np.pi/12.
        nodeXY = np.array([0., 0., -np.cos(theta), np.sin(theta), np.cos(theta), np.sin(theta)]).reshape(-1,2)
        connectivity = np.array([0, 1, 0, 2]).reshape(-1,2)
        fixed = {'XNodes':np.array([1,2]), 'YNodes':np.array([1,2])}
        force = {'nodes':np.array([0]), 'fx':1.E3*torch.tensor([np.cos(beta)]).float(), 'fy':-1.E3*torch.tensor([np.sin(beta)]).float()}

    if(exampleNo == 4):
        exampleName = 'midCantilever'
        nodeXY = np.array([0., -0.5, 1., -0.4, 2., 0., 1., 0.4, 0., 0.5]).reshape(-1,2)
        connectivity = np.array([0, 1, 1, 2, 3, 2, 4, 3, 0, 3, 4, 1]).reshape(-1,2)
        fixed = {'XNodes':np.array([0,4]), 'YNodes':np.array([0,4])}
        force = {'nodes':np.array([2]), 'fx':1.E3*torch.tensor([0.]), 'fy':1.E3*torch.tensor([-4000.])}

    if(exampleNo == 5):
        exampleName = 'tipCantilever'
        nodeXY = np.array([0., -1., 2., -1., 1., 0., 0., 0]).reshape(-1,2)
        connectivity = np.array([0, 1, 2, 1, 3, 2, 0, 2]).reshape(-1,2)
        fixed = {'XNodes':np.array([0,3]), 'YNodes':np.array([0,3])}
        force = {'nodes':np.array([1]), 'fx':1.E3*torch.tensor([0.]), 'fy':1.E3*torch.tensor([-10.])}

    if(exampleNo == 6):
        exampleName = '47TrussAntenna'
        nodeXY = np.array([0., 0., 1., 0., 0., 1., 1., 1., 0., 2., 1., 2., 0., 3.,\
                                                1., 3., 0.25, 3.5, 0.75, 3.5, 0.25, 4., 0.75, 4.,\
                                                0.25, 4.5, 0.75, 4.5, -0.25, 4.75, 1.25, 4.75,\
                                                -0.75, 5., -0.25, 5., 0.25, 5., 0.75, 5., 1.25, 5.,\
                                                1.75, 5.]).reshape(-1,2)
        connectivity = (np.array([1, 3, 1, 4 ,2, 3, 2, 4, 3, 4, 3, 5, 3, 6, 4, 5,\
                                                            4, 6, 5, 6, 5, 7, 5, 8, 6, 7, 6, 8,\
                                                            7, 8, 7, 9, 7, 10, 8, 9, 8, 10, 9, 10, 9, 11,\
                                                            9, 12, 10, 11, 10, 12, 11, 12, 11, 13, 11, 14, \
                                                            12, 13, 12, 14, 13, 14, 13, 15, 13, 19, 13, 20, \
                                                            14, 19, 14, 20, 14, 16, 15, 17, 15, 18, 15, 19,\
                                                            16, 20, 16, 21, 16, 22, 17, 18, 18, 19, 19,\
                                                            20, 20, 21, 21, 22])-1).reshape(-1,2)

        fixed = {'XNodes':np.array([0,1]), 'YNodes':np.array([0,1])}
        force = {'nodes':np.array([16, 21]), 'fx':1.E3*torch.tensor([0., 0.]),\
                                                                 'fy':1.E3*torch.tensor([-10., -10])}
    if(exampleNo == 7):
        exampleName = 'tipCantileverBig'
        nodeXY = 1.*np.array([0., 0., 1., -0.2, 1.5, -0.7, 1.9, -1.2, \
                                             2., -2., 1.6, -2., 0.8, -2., 0., -2]).reshape(-1,2)
        connectivity = (np.array([1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, \
                                                            1, 6, 1, 7, 2, 8, 3, 8, 4, 8])-1).reshape(-1,2)
        fixed = {'XNodes':np.array([0,7]), 'YNodes':np.array([0,7])}
        force = {'nodes':np.array([4]), 'fx':1.E3*torch.tensor([1.]), 'fy':1.E3*torch.tensor([-0.])}

    if(exampleNo == 8):
        exampleName = 'tipCantilever'
        nodeXY = np.array([0., -1., 2., -1., 1., 0., 0., 0., 1.5, -0.4]).reshape(-1,2)
        connectivity = np.array([0, 1, 1, 4, 2, 4, 2, 3, 0, 2, 0, 4]).reshape(-1,2)
        fixed = {'XNodes':np.array([0,3]), 'YNodes':np.array([0,3])}
        force = {'nodes':np.array([1]), 'fx':1.E3*torch.tensor([0.]), 'fy':1.E3*torch.tensor([-10.])}

    if(exampleNo == 9):
        exampleName = 'bridge'
        nodeXY = np.array([0., 0., 1., 0., 2., 0., 3., 0., 2., 1., 1., 1.]).reshape(-1,2)
        connectivity = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 1, 5, 2, 4, 2, 5, 0, 5]).reshape(-1,2)
        fixed = {'XNodes':np.array([0]), 'YNodes':np.array([0,3])}
        force = {'nodes':np.array([1, 2, 4]), 'fx':1.E3*torch.tensor([0., 0., 1000.]), 'fy':1.E3*torch.tensor([-1000., -1000, 0.])}

    if(exampleNo == 10):
        exampleName = 'bridge'
        nodeXY = np.array([0., 0., 1., 0., 2., 0., 3., 0., 2., 1., 1., 1.]).reshape(-1,2)
        connectivity = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 1, 5, 2, 4, 2, 5, 0, 5]).reshape(-1,2)
        fixed = {'XNodes':np.array([0]), 'YNodes':np.array([0,3])}
        force = {'nodes':np.array([1, 2, 4]), 'fx':1.E3*torch.tensor([0., 0., 0.]), 'fy':1.E3*torch.tensor([-1000., -1000, 0.])}

    if(exampleNo == 11):
        exampleName = 'indeterminateTruss'
        nodeXY = np.array([0., 0.,0.5,0.25,1,0.5,1.5,0.25,2,0.,1.5,1.,0.5,1.]).reshape(-1,2)
        connectivity = -1+np.array([1,2,1,7,2,7,2,3,3,7,3,6,3,4,4,6,4,5,5,6,6,7,1,5]).reshape(-1,2)
        fixed = {'XNodes':np.array([0]), 'YNodes':np.array([0,4])}
        force = {'nodes':np.array([5]), 'fx':1.E3*torch.tensor([1.]), 'fy':1.E3*torch.tensor([0.])}

    if(exampleNo == 12):
        exampleName = '3D'
        nodeXY = np.array([0.,0.,0.,3.,0.,0.,0.,0.,2.,3.,0.,2.,1.,1.,1.,2.,1.,1]).reshape(-1,3)
        connectivity = np.array([0,4,2,4,4,5,1,5,3,5]).reshape(-1,2)
        fixed = {'XNodes':np.array([0,1,2,3]), 'YNodes':np.array([0,1,2,3]),'ZNodes':np.array([0,1,2,3])}
        force = {'nodes':np.array([4]), 'fx':1.E0*torch.tensor([1.]), 'fy':0E0*torch.tensor([0.]), 'fz':0.E3*torch.tensor([-0.])}

    if(exampleNo == 13):
        exampleName = '3D1Member'
        nodeXY = np.array([0.,0.,0.,1.,0.,0.]).reshape(-1,3)
        connectivity = np.array([0,1]).reshape(-1,2)
        fixed = {'XNodes':np.array([0]), 'YNodes':np.array([0,1]),'ZNodes':np.array([0,1])}
        force = {'nodes':np.array([1]), 'fx':torch.tensor([1.]), 'fy':0.E0*torch.tensor([1.]), 'fz':0.E3*torch.tensor([-0.])}

    if(exampleNo == 14):
        exampleName = '2D1Member'
        nodeXY = np.array([0.,0.,1.,0.]).reshape(-1,2)
        connectivity = np.array([0,1]).reshape(-1,2)
        fixed = {'XNodes':np.array([0]), 'YNodes':np.array([0,1])}
        force = {'nodes':np.array([1]), 'fx':1*torch.tensor([1.]), 'fy':0.E0*torch.tensor([1.])}




    if(exampleNo == 15):
        exampleName = '2Dunm'
        nodeXY = np.array([0.,0.,40.,0.,40,30,0,30]).reshape(-1,2)
        connectivity = np.array([0,1,1,2,0,2,2,3]).reshape(-1,2)
        fixed = {'XNodes':np.array([0,3]), 'YNodes':np.array([0,1,3])}
        force = {'nodes':np.array([1,2]), 'fx':torch.tensor([20000.,0]), 'fy':torch.tensor([0.,-25000.])}

    if(exampleNo == 16):
        exampleName = '3Dunm'
        nodeXY = np.array([0.,0.,0.,40.,0.,0,40,30,0.,0,30,0]).reshape(-1,3)
        connectivity = np.array([0,1,1,2,0,2,2,3]).reshape(-1,2)
        fixed = {'XNodes':np.array([0,3]), 'YNodes':np.array([0,1,3]), 'ZNodes':np.array([0,1,2,3])}
        force = {'nodes':np.array([1,2]), 'fx':torch.tensor([20000.,0]), 'fy':torch.tensor([0.,-25000.]), 'fz':0*torch.tensor([0.,0.])}

    if(exampleNo == 17):
        exampleName = '3D4member'
        nodeXY = np.array([-1,0,-1,\
                           1,0,-1,\
                           1,0,1,\
                           -1,0,1,\
                           0,2,0]).reshape(-1,3)
        connectivity = np.array([0,4,1,4,2,4,3,4]).reshape(-1,2)
        fixed = {'XNodes':np.array([0,1,2,3]), 'YNodes':np.array([0,1,2,3]), 'ZNodes':np.array([0,1,2,3])}
        force = {'nodes':np.array([4]), 'fx':torch.tensor([0.]), 'fy':torch.tensor([-1e6]), 'fz':0*torch.tensor([0.])}

    if(exampleNo == 18):
        exampleName = '3DAntenna'
        nodeXY = 1e0*np.array([-0.9525, 0, 5.0800,\
                            0.9525,         0,    5.0800,\
                           -0.9525,    0.9525,    2.5400,\
                            0.9525,    0.9525,    2.5400,\
                            0.9525,   -0.9525,    2.5400,\
                           -0.9525,   -0.9525,    2.5400,\
                           -2.5400,    2.5400,         0,\
                            2.5400,    2.5400,         0,\
                            2.5400,   -2.5400,         0,\
                           -2.5400,   -2.5400,         0]).reshape(-1,3)
        connectivity = -1 + np.array([[1,1,2,1,2,2,2,1,1,3,4,3,6,3,6,4,5,4,3,5,6,6,3,4,5],
                                      [2,4,3,5,6,4,5,3,6,6,5,4,5,10,7,9,8,7,8,10,9,10,7,8,9]]).T
        fixed = {'XNodes':np.array([6,7,8,9]), 'YNodes':np.array([6,7,8,9]), 'ZNodes':np.array([6,7,8,9])}
        force = {'nodes':np.array([0,1]), 'fx':torch.tensor([0.,0.]), 'fy':torch.tensor([0.,0.]), 'fz':torch.tensor([-100.,-100.])}

    if(exampleNo == 19):
        exampleName = '3DCantilever'
        alpha = 1;
        beta = 0.5;
        gamma = 0.5;
        nodeXY = np.array([0.,0.,0.,\
                           alpha ,0.,0.,\
                           alpha ,beta,0.,\
                           0.,beta,0.,\
                           0.,0.,gamma,\
                           alpha,0.,gamma,\
                           alpha,beta,gamma,\
                           0.,beta,gamma,\
                           2.0*alpha,0.5*beta,0.5*gamma]).reshape(-1,3)
        connectivity = np.array([0,1,0,2,0,5,\
                                  3,2,3,6,3,1,\
                                      4,1,4,5,4,6,\
                                          7,6,7,2,7,5,\
                                              1,2,1,5,2,6,5,6,\
                                                  1,8,5,8,2,8,6,8]).reshape(-1,2)
        fixed = {'XNodes':np.array([0,3,4,7]), 'YNodes':np.array([0,3,4,7]), 'ZNodes':np.array([0,3,4,7])}
        force = {'nodes':np.array([8]), 'fx':torch.tensor([0.]), 'fy':1e6*torch.tensor([200.]), 'fz':torch.tensor([0.])}

    bc = {'forces':force, 'fixtures':fixed}

    return exampleName, nodeXY, connectivity, bc