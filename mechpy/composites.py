# coding: utf-8

'''
Module to be used for composite material analysis
'''

from __future__ import print_function
from __future__ import division
from numpy import pi, zeros, ones, linspace, arange, array, sin, cos
from numpy.linalg import solve, inv
#from scipy import linalg
import numpy as np
#np.set_printoptions(suppress=False,precision=2)   # suppress scientific notation
np.set_printoptions(precision=4, linewidth=200)
np.set_printoptions(linewidth=200)

import pandas as pd

import sympy as sp
sp.init_printing(wrap_line=False, pretty_print=True)

from pprint import pprint

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,figure,xlim,ylim,title,legend, \
grid, show, xlabel,ylabel, tight_layout
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d  

mpl.rcParams['figure.figsize'] = (8,5)
mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 14


# inline plotting
from IPython import get_ipython
#get_ipython().magic('matplotlib inline')

###disable inline plotting
get_ipython().magic('matplotlib inline')    
    

def import_matprops(mymaterial='T300_5208'):
    '''
    import material properties
    '''

    matprops = pd.read_csv('F:/mechpy/mechpy/compositematerials.csv', index_col=0)

    mat = matprops[mymaterial]
    return mat

def Sf6(E1,E2,nu12,nu23,G12):
    '''transversly isptropic compliance matrix. pg 58 herakovich'''
    S = array([ [1/E1, -nu12/E1, -nu12/E1,0,0,0],
              [-nu12/E1,1/E2,-nu23/E2,0,0,0],
              [-nu12/E1, -nu23/E2, 1/E2,0,0,0],
              [0,0,0,2*(1+nu23)/E2,0,0],
              [0,0,0,0,1/G12,0],
              [0,0,0,0,0,1/G12]])
    return S


def Sf(E1,E2,nu12,G12):
    '''transversely isptropic compliance matrix. pg 58 herakovich'''
    nu21 = E2*nu12/E1
    S = array([[1/E1,    -nu21/E2, 0],
               [-nu12/E1, 1/E2,    0],
               [0,        0,       1/G12]])
    return S    

def S6f(E1,E2,nu12,nu23,G12):
    '''
    transversely isotropic compliance matrix. 
    For transversly isotropic
    E2=E3, nu12=nu13,G12=G13,G23=E2/(2(1+nu23))
    '''
    G23=E2/(2*(1+nu23))
    S6 = array( [[    1/E1, -nu12/E1, -nu12/E1,     0,     0,       0],  
                 [-nu12/E1,     1/E2, -nu23/E2,     0,     0,       0],  
                 [-nu12/E1, -nu23/E2,     1/E2,     0,     0,       0],
                 [     0,        0,        0,      1/G23,  0,       0],
                 [     0,        0,        0,       0,    1/G12,    0],
                 [     0,        0,        0,       0,     0,   1/G12]])    
    return S6
    
def Qf(E1,E2,nu12,G12):
    '''transversly isptropic compliance matrix. pg 58 herakovich'''
    nu21 = E2*nu12/E1
    Q = array([[E1/(1-nu12*nu21),    E2*nu12/(1-nu12*nu21), 0],
               [ E2*nu12/(1-nu12*nu21), E2/(1-nu12*nu21),    0],
               [0,        0,       G12]])
    return Q  

def T61(th):
    '''Stress
    th=ply angle in degrees
    voight notation for stress tranform. sigma1 = T1 @ sigmax'''
    n = sin(th*pi/180)
    m = cos(th*pi/180)
    T1 = array( [[m**2, n**2, 0, 0, 0, 2*m*n],  
                 [n**2, m**2, 0, 0, 0,-2*m*n],  
                 [0,    0,    1, 0, 0, 0],
                 [0,    0,    0, m,-n, 0],
                 [0,    0,    0, n, m, 0],
                 [-m*n, m*n,  0, 0, 0,(m**2-n**2)]])    
    return T1
    
def T62(th):
    '''Strain
    voight notation for strain transform. epsilon1 = T2 @ epsilonx
    th=ply angle in degrees
    '''
    n = sin(th*pi/180)
    m = cos(th*pi/180)
    T2 = array( [[m**2, n**2, 0, 0, 0, m*n],  
                 [n**2, m**2, 0, 0, 0,-m*n],  
                 [0,    0,    1, 0, 0, 0],
                 [0,    0,    0, m,-n, 0],
                 [0,    0,    0, n, m, 0],
                 [-2*m*n, 2*m*n,  0, 0, 0,(m**2-n**2)]])     
    return T2


def T1(th):
    '''Stress Transform for Plane Stress 
    th=ply angle in degrees
    voight notation for stress tranform. sigma1 = T1 @ sigmax
    recall T1(th)**-1 == T1(-th)'''
    n = sin(th*pi/180)
    m = cos(th*pi/180)
    T1 = array( [[m**2, n**2, 2*m*n],  
                 [n**2, m**2,-2*m*n],  
                 [-m*n, m*n,(m**2-n**2)]])    
    return T1
    
def T2(th):
    '''Strain Transform for Plane Stress
    th=ply angle in degrees
    voight notation for strain transform. epsilon1 = T2 @ epsilonx'''
    n = sin(th*pi/180)
    m = cos(th*pi/180)
    T2 = array( [[m**2, n**2, m*n],  
                 [n**2, m**2,-m*n],  
                 [-2*m*n, 2*m*n,  (m**2-n**2)]])     
    return T2

def material_plots():
    '''
    plotting composite properties
    '''
#    plt.rcParams['figure.figsize'] = (10, 8)
#    plt.rcParams['font.size'] = 14
#    plt.rcParams['legend.fontsize'] = 14
    
    get_ipython().magic('matplotlib inline') 
    
    plt.close('all')
    
    mat = import_matprops('T300_5208')
    S = Sf6(mat.E1,mat.E2,mat.nu12,mat.nu23,mat.G12 )    
    C = inv(S)
    theta = arange(-90, 90.1, 0.1) 
    
    C11 = [(inv(T61(th)) @ C @ T62(th))[0,0] for th in theta]
    C22 = [(inv(T61(th)) @ C @ T62(th))[1,1] for th in theta]
    C44 = [(inv(T61(th)) @ C @ T62(th))[3,3] for th in theta]
    C66 = [(inv(T61(th)) @ C @ T62(th))[5,5] for th in theta]
    
    Exbar = zeros(len(theta))
    Eybar = zeros(len(theta))
    Gxybar = zeros(len(theta))

    h = 1      # lamina thickness
    Q = Qf(mat.E1,mat.E2,mat.nu12,mat.G12)

    Qbar = zeros((len(theta),3,3))
    for i,th in enumerate(theta):
        Qbar[i] = solve(T1(th), Q) @ T2(th)
    #Qbar = [solve(T1(th),Q) @ T2(th) for th in theta]

    Qbar11 = Qbar[:,0,0]
    Qbar22 = Qbar[:,1,1]
    Qbar66 = Qbar[:,2,2]
    Qbar12 = Qbar[:,0,1]
    Qbar16 = Qbar[:,0,2]
    Qbar26 = Qbar[:,1,2]  

    Aij = Qbar*h

    # laminate Stiffness
    #     | Exbar    Eybar    Gxybar   |
    # A = | vxybar   vyxbar   etasxbar |
    #     | etaxsbar etaysbar etasybar | 

    # laminate Comnpliance
    aij = zeros((len(theta),3,3))
    for i, _Aij in enumerate(Aij):
        aij[i] = inv(_Aij)   

    # material properties for whole laminate (Daniel, pg183)
    Exbar  = [1/(h*_aij[0,0]) for _aij in aij]
    Eybar  = [1/(h*_aij[1,1]) for _aij in aij]
    Gxybar = [1/(h*_aij[2,2]) for _aij in aij]    
    
    # Global Stress
    s_xy = array([[100], 
                  [10], 
                  [5]])
    
    # local ply stress
    s_12 = np.zeros((3,len(theta)))
    for i,th in enumerate(theta):
        #s_12[:,i] = np.transpose(T1(th) @ s_xy)[0]   # local stresses
        s_12[:,[i]] = T1(th) @ s_xy   
    
    
    # Plotting
    figure()#, figsize=(10,8))
    plot(theta, C11, theta, C22, theta, C44, theta, C66)
    legend(['$\overline{C}_{11}$','$\overline{C}_{22}$', '$\overline{C}_{44}$', '$\overline{C}_{66}$'])
    title('Transversly Isotropic Stiffness properties of carbon fiber T300_5208')
    xlabel("$\Theta$")
    ylabel('$\overline{C}_{ii}$, ksi')
    grid()

    figure()#, figsize=(10,8))
    plot(theta, Exbar, label = r"Modulus: $E_x$")
    plot(theta, Eybar, label = r"Modulus: $E_y$")
    plot(theta, Gxybar, label = r"Modulus: $G_{xy}$")
    title("Constitutive Properties in various angles")
    xlabel("$\Theta$")
    ylabel("modulus, GPa")
    legend()
    grid()
    
    figure()#,figsize=(10,8))
    plot(theta, s_12[0,:], label = '$\sigma_{11},ksi$' )
    plot(theta, s_12[1,:], label = '$\sigma_{22},ksi$' )
    plot(theta, s_12[2,:], label = '$\sigma_{12},ksi$' )
    legend(loc='lower left')
    xlabel("$\Theta$")
    ylabel("Stress, ksi")
    grid()

    # plot theta as a function of time
    figure()#,figsize=(10,8))
    plot(theta,Qbar11, label = "Qbar11")
    plot(theta,Qbar22, label = "Qbar22")
    plot(theta,Qbar66, label = "Qbar66")
    legend(loc='lower left')
    xlabel("$\Theta$")
    ylabel('Q')
    grid()

    # plot theta as a function of time
    figure()#,figsize=(10,8))
    plot(theta,Qbar12, label = "Qbar12")
    plot(theta,Qbar16, label = "Qbar16")
    plot(theta,Qbar26, label = "Qbar26")
    legend(loc='lower left')
    xlabel("$\Theta$")
    ylabel('Q')
    grid()
    
    show()

   
def laminate_gen(lamthk=1.5, symang=[45,0,90], plyratio=2.0, matrixlayers=False, balancedsymmetric=True):
    '''
    ## function created to quickly create laminates based on given parameters
    lamthk=1.5    # total #thickness of laminate
    symang = [45,0,90, 30]  #symmertic ply angle
    plyratio=2.0  # lamina/matrix ratio
    matrixlayers=False  # add matrix layers between lamina plys
    nonsym=False    # symmetric
    mat = material type, as in different plies, matrix layer, uni tapes, etc
    #ply ratio can be used to vary the ratio of thickness between a matrix ply
         and lamina ply. if the same thickness is desired, plyratio = 1, 
         if lamina is 2x as thick as matrix plyratio = 2
    '''
    if matrixlayers:
        nl = (len(symang)*2+1)*2
        nm = nl-len(symang)*2
        nf = len(symang)*2
        tm = lamthk / (plyratio*nf + nm)
        tf = tm*plyratio
        theta = zeros(nl//2)
        mat = 2*ones(nl//2)  #  orthotropic fiber and matrix = 1, isotropic matrix=2, 
        mat[1:-1:2] = 1   #  [2 if x%2 else 1 for x in range(nl//2) ]
        theta[1:-1:2] = symang[:]  # make a copy
        thk = tm*ones(nl//2)
        thk[2:2:-1] = tf
        lamang = list(symang) + list(symang[::-1])
        theta = list(theta) + list(theta[::-1])
        mat = list(mat) + list(mat[::-1])
        thk = list(thk) + list(thk[::-1])  
    else: # no matrix layers, ignore ratio
        if balancedsymmetric:
            nl = len(symang)*2
            mat = list(3*np.ones(nl)) 
            thk = list(lamthk/nl*np.ones(nl))
            lamang = list(symang) + list(symang[::-1])
            theta = list(symang) + list(symang[::-1])
        else:            
            nl = len(symang)
            mat =[1]*nl
            thk = list(lamthk/nl*np.ones(nl))
            lamang = symang[:]
            theta = symang[:]

    return thk,theta,mat,lamang



def laminate():
    '''
    code to compute composite properties, applied mechanical and thermal loads
    and stress and strain
    
    general outline for computing elastic properties of composites
    
    1) Determine engineering properties of unidirectional laminate. E1, E2, nu12, G12
    2) Calculate ply stiffnesses Q11, Q22, Q12, Q66 in the principal/local coordinate system
    3) Determine Fiber orientation of each ply
    4) Calculate the transformed stiffness Qxy in the global coordinate system
    5) Determine the through-thicknesses of each ply
    6) Determine the laminate stiffness Matrix (ABD)
    7) Calculate the laminate compliance matrix by inverting the ABD matrix
    8) Calculate the laminate engineering properties
    
    TODO -    add failure criteria
	TODO -    add multiple material properties for a laminate
    
    # Stress Strain Relationship for a laminate, with Q=reduced stiffness matrix
    |sx | |Qbar11 Qbar12 Qbar16| |ex +z*kx |
    |sy |=|Qbar12 Qbar22 Qbar26|=|ey +z*ky |
    |sxy| |Qbar16 Qbar26 Qbar66| |exy+z*kxy|
    
    # Herakovich pg 84
    Qbar =  inv(T1) @ Q @ T2 == solve(T1, Q) @ T2
    
    transformation reminders - see Herakovich for details
    sig1 = T1*sigx
    sigx = inv(T1)*sig1
    eps1 = T2*epsx
    epsx = inv(T2)*epsx
    sigx = inv(T1)*Q*T2*epsx
    Qbar = inv(T1)*Q*T2
    Sbar = inv(T2)*inv(Q)*T2
    '''
    
    #==========================================================================
    # Initialize
    #==========================================================================
    get_ipython().magic('matplotlib') 
    plt.close('all')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 13
    #plt.rcParams['legend.fontsize'] = 14
    
    #==========================================================================
    # Import Material Properties
    #==========================================================================
    mat  = import_matprops('graphite-polymer_SI')  # Hyer  #import_matprops('T300_5208') #
    #mat  = import_matprops('T300_5208')  # Herakovich
    alpha = array([[mat.alpha1], [mat.alpha2], [0]])
    
    W =   0.25  # plate width
    L =  .125           # laminate length  
    plyangle = [30,-30,0,0,-30,30]
    laminatethk =   zeros(len(plyangle)) + mat.plythk  # ply thicknesses
    nply = len(laminatethk) # number of plies
    H =   mat.plythk*nply # plate thickness
    #    area = W*H
    z = zeros(nply+1)
    zmid = zeros(nply)
    z[0] = -H/2
    for i in range(nply):
        z[i+1] = z[i] + laminatethk[i]
        zmid[i] = z[i] + laminatethk[i]/2
    
    #==========================================================================
    # ABD Matrix Compute
    #==========================================================================
    # Reduced stiffness matrix for a plane stress ply in principal coordinates
    # calcluating Q from the Compliance matrix may cause cancellation errors
    Q = Qf(mat.E1,mat.E2,mat.nu12,mat.G12)

    A = zeros((3,3)); B = zeros((3,3)); D = zeros((3,3))  
    for i in range(nply):  # = nply
        Qbar = solve(T1(plyangle[i]), Q) @ T2(plyangle[i])
        A += Qbar*(z[i+1]-z[i])
        # coupling  stiffness
        B += (1/2)*Qbar*(z[i+1]**2-z[i]**2)
        # bending or flexural laminate stiffness relating moments to curvatures
        D += (1/3)*Qbar*(z[i+1]**3-z[i]**3)      
    
    # laminate stiffness matrix
    ABD = zeros((6,6))
    ABD[0:3,0:3] = A
    ABD[0:3,3:6] = B
    ABD[3:6,0:3] = B
    ABD[3:6,3:6] = D
 
    # laminatee compliance
    abcd = inv(ABD)
    a = abcd[0:3,0:3]
    
    #==========================================================================
    # Laminate Properties    
    #==========================================================================
    
    # effective laminate shear coupling coefficients
    etasxbar = a[0,2]/a[2,2]
    etasybar = a[1,2]/a[2,2]
    etaxsbar = a[2,0]/a[0,0]
    etaysbar = a[2,1]/a[1,1]
    
    # laminate engineer properties
    Exbar  = 1 / (H*a[0,0])
    Eybar  = 1 / (H*a[1,1])
    Gxybar = 1 / (H*a[2,2])
    nuxybar = -a[0,1]/a[0,0]
    nuyxbar = -a[0,1]/a[1,1]
    

    # --------------------- Double Check ---------------------
#    # Laminate compliance matrix
#    LamComp = array([ [1/Exbar,       -nuyxbar/Eybar,  etasxbar/Gxybar],
#                      [-nuxybar/Exbar,  1/Eybar ,       etasybar/Gxybar],
#                      [etaxsbar/Exbar, etaysbar/Eybar, 1/Gxybar]] ) 
#    # Daniel pg 183
#    # combines applied loads and applied strains
#    strain_laminate = LamComp @ Nxyzapplied[:3]/H + strainxyzapplied[:3]
#    Nxyz = A @ strain_laminate
#    stress_laminate = Nxyz/H
    # --------------------------------------------------------
    

    #==========================================================================
    #  Applied Loads 
    #==========================================================================
    # either apply strains or loads 
            #               Nx Ny  Nxy  Mx  My Mxy 
    NMbarapp =      array([[0],[0],[0],[12.84],[0],[0]])
    #                       ex ey exy  kx  ky kxy
    epsilonbarapp = array([[0],[0],[0],[0],[0],[0]]) 
    
    NMbarapptotal = NMbarapp + ABD@epsilonbarapp
    #==========================================================================
    # Thermal Loads  
    #==========================================================================
    '''
    if the material is isotropic and unconstrained, then no thermal stresses 
        will be experienced. If there are constraints, then the material will experience
        thermally induced stresses. As with orthotropic materials, various directions will have 
        different stresses, and when stacked in various orientations, stresses can be 
        unintuitive and complicated. Global Thermal strains are subtracted from applied strains 
    # 1) determine the free unrestrained thermal strains in each layer, alphabar
    '''
    
    Ti = 150   # initial temperature (C)
    Tf = 0 # final temperature (C)
    dT = Tf-Ti 
    
    Nhatth= zeros((3,1))  # unit thermal force in global CS
    Mhatth = zeros((3,1)) # unit thermal moment in global CS
    alphabar = zeros((3,nply))    # global ply CTE 
    for i in range(nply):  # = nply
        Qbar = solve(T1(plyangle[i]), Q) @ T2(plyangle[i])
        alphabar[:,[i]] = solve(T2(plyangle[i]),  alpha)
        #alphabar[:,[i]] = inv(T2(plyangle[i])) @ alpha # Convert to global CS    
        Nhatth += Qbar @ (alphabar[:,[i]])*(z[i+1] - z[i]) # Hyer method for calculating thermal unit loads
        Mhatth += 0.5*Qbar@(alphabar[:,[i]])*(z[i+1]**2-z[i]**2)   
        
    NMhatth = np.vstack((Nhatth,Mhatth))
    NMbarth = NMhatth*dT # resultant thermal loads    
    
    # Laminate CTE
    epsilonhatth = abcd@NMhatth # laminate CTE
    
    # applied loads and thermal loads
    epsilonbarapp = abcd @ NMbarapptotal
    epsilonbarth  = abcd @ NMbarth  # resultant thermal strains
    epsilonbartotal = epsilonbarapp + epsilonbarth
    
    # Composite respone from applied mechanical loads and strains. Average
    # properties only. Used to compare results from tensile test.
    #epsilon_laminate = abcd@NMbarapptotal
    #sigma_laminate = ABD@epsilon_laminate/H    
    
    epsilon_laminate = epsilonbartotal[:]
    sigma_laminate = ABD@epsilonbartotal/H    
    alpha_laminate = a@Nhatth

    # determine thermal load and applied loads or strains Hyer pg 435,452
    Nx = NMbarapptotal[0,0]*W # units kiloNewtons, total load as would be applied in a tensile test
    Ny = NMbarapptotal[1,0]*L # units kN 

    #==========================================================================
    # Thermal and mechanical local and global stresses at the ply interface
    #==========================================================================
    # Declare variables for plotting
    epsilon_app         = zeros((3,2*nply))    
    sigma_app           = zeros((3,2*nply))    
    epsilonbar_app      = zeros((3,2*nply))
    sigmabar_app        = zeros((3,2*nply))
    epsilon_th          = zeros((3,2*nply))
    sigma_th            = zeros((3,2*nply))    
    epsilonbar_th       = zeros((3,2*nply))
    sigmabar_th         = zeros((3,2*nply))
    epsilon             = zeros((3,2*nply))    
    epsilonbar          = zeros((3,2*nply))
    sigma               = zeros((3,2*nply))    
    sigmabar            = zeros((3,2*nply))
    
    for i,k in enumerate(range(0,2*nply,2)):
        # stress is calcuated at top and bottom of each ply
        Qbar = solve(T1(plyangle[i]), Q) @ T2(plyangle[i])

         # Global stresses and strains, applied load only
        epsbarapp1 = epsilonbarapp[0:3] + z[i]*epsilonbarapp[3:7]
        epsbarapp2 = epsilonbarapp[0:3] + z[i+1]*epsilonbarapp[3:7]
        sigbarapp1 = Qbar @ epsbarapp1
        sigbarapp2 = Qbar @ epsbarapp2
        # Local stresses and strains, appplied load only
        epsapp1 = T2(plyangle[i]) @ epsbarapp1
        epsapp2 = T2(plyangle[i]) @ epsbarapp2
        sigapp1 = Q @ epsapp1
        sigapp2 = Q @ epsapp2
        # Interface Stresses and Strains
        epsilon_app[:,k:k+2]    = np.column_stack((epsapp1,epsapp2))
        epsilonbar_app[:,k:k+2] = np.column_stack((epsbarapp1,epsbarapp2))
        sigma_app[:,k:k+2]      = np.column_stack((sigapp1,sigapp2))
        sigmabar_app[:,k:k+2]   = np.column_stack((sigbarapp1,sigbarapp2))

        # Global stress and strains, thermal loading only
        epsbarth1 = epsilonbarth[0:3] + z[i]*epsilonbarth[3:7]   - dT*alphabar[:,[i]]
        epsbarth2 = epsilonbarth[0:3] + z[i+1]*epsilonbarth[3:7] - dT*alphabar[:,[i]]
        sigbarth1 = Qbar @ epsbarth1
        sigbarth2 = Qbar @ epsbarth2   
        
        # Local stress and strains, thermal loading only
        epsth1 = T2(plyangle[i]) @ epsbarth1
        epsth2 = T2(plyangle[i]) @ epsbarth2
        sigth1 = Q @ epsth1
        sigth2 = Q @ epsth2
        
        # Interface Stresses and Strains
        epsilon_th[:,k:k+2]    = np.column_stack((epsth1,epsth2))
        epsilonbar_th[:,k:k+2] = np.column_stack((epsbarth1+dT*alphabar[:,[i]],epsbarth2+dT*alphabar[:,[i]])) # remove the local thermal loads for plotting. only use local thermal strains for calculating stress
        sigma_th[:,k:k+2]      = np.column_stack((sigth1,sigth2))
        sigmabar_th[:,k:k+2]   = np.column_stack((sigbarth1,sigbarth2))       
        
        # TOTAL global stresses and strains, applied and thermal
        epsbar1 = epsbarapp1 + epsbarth1
        epsbar2 = epsbarapp2 + epsbarth2
        sigbar1 = Qbar @ epsbar1
        sigbar2 = Qbar @ epsbar2
        # TOTAL local stresses and strains , applied and thermal
        eps1 = T2(plyangle[i]) @ epsbar1
        eps2 = T2(plyangle[i]) @ epsbar2
        sig1 = Q @ eps1
        sig2 = Q @ eps2
        # Interface Stresses and Strains
        epsilon[:,k:k+2]     = np.column_stack((eps1,eps2))
        epsilonbar[:,k:k+2]  = np.column_stack((epsbar1+dT*alphabar[:,[i]],epsbar2+dT*alphabar[:,[i]])) # remove the local thermal loads for plotting. only use local thermal strains for calculating stress
        sigma[:,k:k+2]       = np.column_stack((sig1,sig2))
        sigmabar[:,k:k+2]    = np.column_stack((sigbar1,sigbar2))   
            
    #==========================================================================
    # Printing Results    
    #==========================================================================
    print('--------------- laminate1 Stress analysis of fibers----------')
    print('plyangles'); print(plyangle)
    print('ply layers') ; print(z)
    print('alpha')
    print(alpha)
    print('ABD=');print(ABD)   
    print('Ex=   %0.2E'%Exbar)
    print('Ey=   %0.2E'%Eybar)
    print('nuxy= %0.2E'%nuxybar)
    print('Gxy=  %0.2E'%Gxybar)
    print('alpha_laminate')
    print(alpha_laminate)
    print('epsilon_laminate')
    print(epsilon_laminate)
    print('NMhatth')    
    print(NMhatth)    
    print('sigma_laminate')
    print(sigma_laminate)
    print('epsilon_th')   
    print(epsilon_th) 
    print('epsilonbar_th')
    print(epsilonbar_th)
    print('sigma_th')
    print(sigma_th)
    print('sigmabar_th')       
    print(sigmabar_th)    
    print('epsilon_app')
    print(epsilon_app)
    print('epsilonbar_app')
    print(epsilonbar_app)
    print('NMbarapp')
    print(NMbarapp)
    print('epsilon')   
    print(epsilon) 
    print('epsilonbar')
    print(epsilonbar)
    print('sigma')
    print(sigma)
    print('sigmabar')       
    print(sigmabar)
            
    #==========================================================================
    # Plotting
    #==========================================================================
    zplot = zeros(2*nply)
    for i,k in enumerate(range(0,2*nply,2)):  # = nply
        zplot[k:k+2] = z[i:i+2]  
        
    #legendlab = ['total','thermal','applied','laminate']
    # global stresses and strains
    mylw = 1.5 #linewidth

    # Global Stresses and Strains
    f1, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2,3, sharex='row', sharey=True)
    f1.canvas.set_window_title('Global Stress and Strain of %s laminate' % (plyangle))
    stresslabel = ['$\sigma_x,\ ksi$','$\sigma_y,\ ksi$','$\\tau_{xy},\ ksi$']
    strainlabel = ['$\epsilon_x$','$\epsilon_y$','$\gamma_{xy}$']
    
    for i,ax in enumerate([ax1,ax2,ax3]):
        ## the top axes
        ax.set_ylabel('thickness,z')
        ax.set_xlabel(strainlabel[i])
        #ax.set_title(' Ply Strain at $\epsilon=%f$' % (epsxapp*100))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(1,4))  # scilimits=(-2,2))
        
        ax.plot(epsilonbar[i,:],     zplot, color='blue', lw=mylw, label='total')
        ax.plot(epsilonbar_th[i,:],  zplot, color='red', lw=mylw, alpha=0.75, linestyle='--',  label='thermal')
        ax.plot(epsilonbar_app[i,:], zplot, color='green', lw=mylw, alpha=0.75,linestyle='-.', label='applied') 
        ax.plot([epsilon_laminate[i], epsilon_laminate[i]],[np.min(z) , np.max(z)], color='black', lw=mylw, label='laminate') 
        ax.grid(True)  
        #ax.set_xticks(linspace( min(ax.get_xticks()) , max(ax.get_xticks()) ,6))           
    
    for i,ax in enumerate([ax4,ax5,ax6]):
        ax.set_ylabel('thickness,z')
        ax.set_xlabel(stresslabel[i])
        #ax.set_title(' Ply Stress at $\sigma=%f$' % (epsxapp*100))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,3)) # scilimits=(-2,2))
        
        ax.plot(sigmabar[i,:],     zplot, color='blue', lw=mylw, label='total')
        ax.plot(sigmabar_th[i,:], zplot, color='red', lw=mylw, alpha=0.75,linestyle='--', label='thermal')
        ax.plot(sigmabar_app[i,:], zplot, color='green', lw=mylw, alpha=0.75,linestyle='-.', label='applied')  
        ax.plot([sigma_laminate[i], sigma_laminate[i]],[np.min(z) , np.max(z)], color='black', lw=mylw, label='laminate') 
        ax.grid(True)

    leg = legend(fancybox=True) ; leg.get_frame().set_alpha(0.3)     
    tight_layout() 
    #mngr = plt.get_current_fig_manager() ; mngr.window.setGeometry(50,50,800, 500)
    f1.show()             
    #plt.savefig('global-stresses-strains.png')    

    ### Local Stresses and Strains
    f2, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2,3, sharex='row', sharey=True)
    f2.canvas.set_window_title('Local Stress and Strain of %s laminate' % (plyangle))
    stresslabel = ['$\sigma_1,\ ksi$','$\sigma_2,\ ksi$','$\\tau_{12},\ ksi$']
    strainlabel = ['$\epsilon_1$','$\epsilon_2$','$\gamma_{12}$']
    
    for i,ax in enumerate([ax1,ax2,ax3]):
        ## the top axes
        ax.set_ylabel('thickness,z')
        ax.set_xlabel(strainlabel[i])
        #ax.set_title(' Ply Strain at $\epsilon=%f$' % (epsxapp*100))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(1,4))  # scilimits=(-2,2))
        
        ax.plot(epsilon[i,:],     zplot, color='blue', lw=mylw, label='total')
        ax.plot(epsilon_th[i,:], zplot, color='red', lw=mylw, alpha=0.75,linestyle='--', label='thermal')
        ax.plot(epsilon_app[i,:], zplot, color='green', lw=mylw, alpha=0.75,linestyle='-.', label='applied')   
        ax.plot([epsilon_laminate[i], epsilon_laminate[i]],[np.min(z) , np.max(z)], color='black', lw=mylw, label='laminate')
        ax.grid(True)
                          
    for i,ax in enumerate([ax4,ax5,ax6]):
        ax.set_ylabel('thickness,z')
        ax.set_xlabel(stresslabel[i])
        #ax.set_title(' Ply Stress at $\sigma=%f$' % (epsxapp*100))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,3)) # scilimits=(-2,2))
        
        ax.plot(sigma[i,:],     zplot, color='blue', lw=mylw, label='total')
        ax.plot(sigma_th[i,:], zplot, color='red', lw=mylw, alpha=0.75,linestyle='--', label='thermal')
        ax.plot(sigma_app[i,:], zplot, color='green', lw=mylw, alpha=0.75,linestyle='-.', label='applied')   
        ax.plot([sigma_laminate[i], sigma_laminate[i]],[np.min(z) , np.max(z)], color='black', lw=mylw, label='laminate')   
        ax.grid(True)
    
    leg = legend(fancybox=True) ; leg.get_frame().set_alpha(0.3)   
    tight_layout() 
    #mngr = plt.get_current_fig_manager() ; mngr.window.setGeometry(850,50,800, 500)
    f2.show()
    #plt.savefig('local-stresses-strains.png')
   
    ### warpage
    res = 100
    Xplt,Yplt = np.meshgrid(np.linspace(-W/2,W/2,res), np.linspace(-L/2,L/2,res))
    epsx = epsilon_laminate[0,0]
    epsy = epsilon_laminate[1,0]
    epsxy = epsilon_laminate[2,0]
    kapx = epsilon_laminate[3,0]
    kapy = epsilon_laminate[4,0]
    kapxy = epsilon_laminate[5,0]
    ### dispalcement
    w = -0.5*(kapx*Xplt**2 + kapy*Yplt**2 + kapxy*Xplt*Yplt)
    u = epsx*Xplt  # pg 451 hyer
    fig = plt.figure('plate-warpage')
    ax = fig.gca(projection='3d')
    ax.plot_surface(Xplt, Yplt, w+zmid[0], cmap=cm.jet, alpha=0.3)
    ###ax.auto_scale_xyz([-(W/2)*1.1, (W/2)*1.1], [(L/2)*1.1, (L/2)*1.1], [-1e10, 1e10])
    ax.set_xlabel('plate width,y-direction,in')
    ax.set_ylabel('plate length,x-direction, in')
    ax.set_zlabel('warpage,in')
    #ax.set_zlim(-0.01, 0.04)
    #mngr = plt.get_current_fig_manager() ; mngr.window.setGeometry(450,550,600, 450)
    plt.show()
    #plt.savefig('plate-warpage')   
   
    
if __name__=='__main__':
    
    
    #material_plots()
    laminate1()


'''CODE GRAVEYARD

def failure_envelope():
    # failure envelopes

    # max stress criteria
    # 1 direction in first row
    # 2 direction in second row
    
    # failure strength in compression
    #Fc = matrix([[-1250.0, -600.0],         
    #            [-200.0,  -120.0]]) # ksi
    #            
    ##failure strength in tension        
    #Ft =  matrix([[1500, 1000]
    #              [50,     30]]) # ksi
    #              
    ##Failure strength in shear              
    #Fs = matrix( [100,    70] ) # Shear
    
    Fc1 = [-1250, -600] # Compression 1 direction
    Fc2 = [-200,  -120] # Compression 2 direction
    Ft1 = [1500, 1000]  # Tension 1 direction
    Ft2 = [50,     30]  # Tension 2 direction
    Fs =  [100,    70]   # Shear
    
    # F1 = Ft(1);
    # F2 = Ft(1);
    # F6 = Fs(1);
    
    for c in range(2):# mattype
        factor = 1.9
        # right
        plot( [Ft1[c], Ft1[c]] , [Fc2[c], Ft2[c]])
        
        # left
        plot( [Fc1[c], Fc1[c]] , [Fc2[c], Ft2[c]])
        # top
        plot( [Fc1[c], Ft1[c]] , [Ft2[c], Ft2[c]])
        # bottom
        plot( [Fc1[c], Ft1[c]]  , [Fc2[c], Fc2[c]])
        # center horizontal
        plot( [Fc1[c], Ft1[c]]  , [0, 0])
        # center vertical
        plot( [0, 0]            , [Fc2[c], Ft2[c]])
        
        #xlim([min(Fc1) max(Ft1)]*factor)
        #ylim([min(Fc2) max(Ft2)]*factor)
        xlabel('$\sigma_1,ksi$')
        ylabel('$\sigma_2,ksi$')
        title('failure envelope with Max-Stress Criteria')   

'''