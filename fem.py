import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from jax.flatten_util import ravel_pytree

class fe_solver_2D():
    tol = 1e-5
    itermax = 20
    n_print = 5

    def __init__(self, node_X, node_x, elements, DOF_fmap, const_model):
        self.node_X         = node_X
        self.node_x         = node_x
        self.elements       = elements
        self.n_node         = len(node_X)
        self.n_elem         = len(elements)
        self.DOF_fmap       = DOF_fmap
        self.const_model    = const_model

    def Nvec(self, xi,eta):
        return 0.25*np.array([(1-xi)*(1-eta),
                              (1+xi)*(1-eta),
                              (1+xi)*(1+eta),
                              (1-xi)*(1+eta)])

    def dNvecdxi(self, xi,eta):
        return 0.25*np.array([[(-1)*(1-eta),   (+1)*(1-eta),   (+1)*(1+eta),   (-1)*(1+eta)],\
                              [(-1)*(1- xi),   (-1)*(1+ xi),   (+1)*(1+ xi),   (+1)*(1- xi)]])

    def assembleRRKK(self):
        node_X      = self.node_X
        elements    = self.elements
        n_node      = self.n_node
        n_elem      = self.n_elem
        node_x      = self.node_x
        const_model = self.const_model

        # assemble total residual 
        RR = np.zeros(n_node*2)
        # assemble the total tangent 
        KK = np.zeros((n_node*2,n_node*2))
        # loop over elements
        for ei in range(n_elem):
            # initialize the residual for this element
            Re = np.zeros((8))
            # initialize the tangent for this element
            Ke = np.zeros((8,8))

            # nodes that make up this element 
            node_ei = elements[ei]
            # reference coordinates of the nodes making up this element (init to zero, fill in a loop)
            node_X_ei = np.zeros((4,2))
            # deformed coordinates of the nodes making up this element (init to zero, fill in a loop)
            node_x_ei = np.zeros((4,2))
            for ni in range(4):
                node_X_ei[ni] = node_X[node_ei[ni]]
                node_x_ei[ni] = node_x[node_ei[ni]]
            
            elem_X = np.mean(node_X_ei, axis=0)
            elem_x = np.mean(node_x_ei, axis=0)
            #print(node_x_ei)
            # also, do a proper integration with four integration points 
            # Loop over integration points
            # location and weight of integration points 
            IP_xi = np.array([[-1./np.sqrt(3),-1./np.sqrt(3)],[+1./np.sqrt(3),-1./np.sqrt(3)],\
                            [+1./np.sqrt(3),+1./np.sqrt(3)],[-1./np.sqrt(3),+1./np.sqrt(3)]])
            IP_wi = np.array([1.,1.,1.,1.])
            for ip in range(4):
                xi  = IP_xi[ip,0]
                eta = IP_xi[ip,1]
                wi = IP_wi[ip]
                # eval shape functions 
                Ns = self.Nvec(xi,eta)
                # eval the isoparametric map for the reference and deformed points corresponding to xi,eta = 0
                X = np.zeros((2))
                x = np.zeros((2))
                for ni in range(4):
                    X += Ns[ni]*node_X_ei[ni]
                    x += Ns[ni]*node_x_ei[ni]

                # evaluate the Jacobians, first derivative of shape functions with respect to xi space then Jacobians 
                dNsdxi = self.dNvecdxi(xi,eta)
                dXdxi = np.zeros((2,2))
                dxdxi = np.zeros((2,2))
                for ni in range(4):
                    dXdxi += np.outer(node_X_ei[ni],dNsdxi[:,ni])
                    dxdxi += np.outer(node_x_ei[ni],dNsdxi[:,ni])
                # get gradient of basis function with respect to X using inverse jacobian 
                JinvT = np.linalg.inv(dXdxi).transpose()
                dNsdX = np.dot(JinvT,dNsdxi)

                # get gradient of basis function with respect to x using inverse jacobian, the other one 
                jinvT = np.linalg.inv(dxdxi).transpose()
                dNsdx = np.dot(jinvT,dNsdxi)

                # get the deformation gradient 
                F = np.zeros((2,2))
                for ni in range(4):
                    F += np.outer(node_x_ei[ni],dNsdX[:,ni])

                # sigma = const_model.sigma(F, node_X_ei[ni], node_x_ei[ni])
                sigma = const_model.sigma(F, elem_X, elem_x)
                
                # compute the variation of the symmetric velocity gradient by moving one node and one component
                # of that node at a time, except if the node is on the boundary in which case no variation is allowed
                for ni in range(4): # Loop over nodes of the element
                    for ci in range(2): # Loop over components (x,y)
                        deltav = np.zeros((2))
                        deltav[ci] = 1
                        gradx_v = np.outer(deltav,dNsdx[:,ni])
                        deltad = 0.5*(gradx_v + gradx_v.transpose())
                        Re[ni*2+ci] += wi*np.linalg.det(dxdxi)*np.tensordot(sigma,deltad)

                        RR[node_ei[ni]*2+ci] += wi*np.linalg.det(dxdxi)*np.tensordot(sigma,deltad)
                        
                        ## 2 more for loops for the increment Delta u
                        for nj in range(4):
                            for cj in range(2):
                                Deltau = np.zeros((2))
                                Deltau[cj]=1
                                gradx_Du = np.outer(Deltau,dNsdx[:,nj])
                                Deltaeps = 0.5*(gradx_Du + gradx_Du.transpose())
                                
                                ## ELEMENT TANGENT
                                # Initial stress component (also called geometric component) is 
                                # sigma: (gradDeltau^T gradv)
                                Kgeom = np.tensordot(sigma,np.dot(gradx_Du.transpose(),gradx_v))
                                # Material component, need to put things in voigt notation for easy computation
                                deltad_voigt = np.array([deltad[0,0],deltad[1,1],2*deltad[0,1]])
                                Deltaeps_voigt = np.array([Deltaeps[0,0],Deltaeps[1,1],2*Deltaeps[0,1]])
                                
                                # D = const_model.D(F, node_X_ei[ni], node_x_ei[ni])
                                D = const_model.D(F, elem_X, elem_x)
                                Kmat = np.dot(Deltaeps_voigt,np.dot(D,deltad_voigt))
                                # add to the corresponding entry in Ke and dont forget other parts of integral
                                Ke[ni*2+ci,nj*2+cj] += wi*np.linalg.det(dxdxi)*(Kgeom+Kmat)
                                # assemble into global 
                                KK[node_ei[ni]*2+ci,node_ei[nj]*2+cj] += wi*np.linalg.det(dxdxi)*(Kgeom+Kmat)
                                
        return RR,KK

    def solve(self):
        DOF_fmap = self.DOF_fmap
        n_node = self.n_node
    
        res = 5
        iter = 0
        print("Solving...")
        while res>self.tol and iter<self.itermax:
            RR,KK = self.assembleRRKK()
            # for the increment not all of the KK is needed because some nodes have essential BC
            RRdof = RR[DOF_fmap.flatten()>=0]
            KKdof = KK[DOF_fmap.flatten()>=0]
            KKdof = KKdof[:,DOF_fmap.flatten()>=0]
            res = np.linalg.norm(RRdof)
            incr_u = -np.linalg.solve(KKdof,RRdof)
            j = 0
            for i in range(n_node):
                if DOF_fmap[i,0] >= 0:
                    self.node_x[i,0]+= incr_u[j]
                    j+= 1
                if DOF_fmap[i,1] >= 0:
                    self.node_x[i,1]+= incr_u[j]
                    j+= 1
            iter +=1
            if iter%self.n_print == 0:
                print('iter {},   residual = {}'.format(iter, res))
        if res<self.tol:
            print("Solution converged!")
        elif iter >= self.itermax:
            print("Reached maximum number of iterations without converging. Either increase itermax or decrease tolerance.")
        
        # Now that the solution has (hopefully) converged, return the deformed coordinates, deformation gradient and stress at center 
        n_elem      = self.n_elem
        node_X      = self.node_X
        node_x      = self.node_x
        elements    = self.elements
        const_model = self.const_model
        F     = np.zeros([n_elem, 2, 2])
        sigma = np.zeros([n_elem, 2, 2])
        for i in range(n_elem):
            xi = 0
            eta = 0
            n1 = elements[i,0]
            n2 = elements[i,1]
            n3 = elements[i,2]
            n4 = elements[i,3]
            dNsdxi = self.dNvecdxi(xi,eta)
            dXdxi = np.outer(node_X[n1],dNsdxi[:,0])+np.outer(node_X[n2],dNsdxi[:,1])\
                    +np.outer(node_X[n3],dNsdxi[:,2])+np.outer(node_X[n4],dNsdxi[:,3])
            # get gradient of basis function with respect to X using inverse jacobian 
            JinvT = np.linalg.inv(dXdxi).transpose()
            dNsdX = np.dot(JinvT,dNsdxi)
            # get the deformation gradient 
            F[i] = np.outer(node_x[n1],dNsdX[:,0])+np.outer(node_x[n2],dNsdX[:,1])\
                +np.outer(node_x[n3],dNsdX[:,2])+np.outer(node_x[n4],dNsdX[:,3])
            sigma[i] = const_model.sigma(F[i], 0.25*(node_X[n1] + node_X[n2] + node_X[n3] + node_X[n4]))
        return self.node_x, F, sigma
    
class neoHookean_const_model(): # Neo Hookean by default.
    def __init__(self, c1 = 1.0):
        self.c1 = c1

    def sigma(self, F, node_X=None, node_x=None):
        c1 = self.c1
        J = np.linalg.det(F)
        b = F @ F.T
        p = 2*c1/J**2
        return -p*np.eye(2) + 2*c1*b
    
    def D(self, F, node_X=None, node_x=None):
        """
        D: spatial tangent moduli (cc) in Voigt notation

        If you only have CC, you can obtain cc using:
            cc = np.einsum('ij,kl,mn,op,jlnp->ikmo', F, F, F, F, CC)

        Then, cc can be cast into Voigt form using:
            D = np.zeros((3,3))
            Itoi = [0,1,0]
            Itoj = [0,1,1]
            for I in range(3):
                for J in range(3):
                    i = Itoi[I]
                    j = Itoj[I]
                    k = Itoi[J]
                    l = Itoj[J]
                    D[I,J] = cc[i,j,k,l]
        """

        c1 = self.c1
        J = np.linalg.det(F)
        p = 2*c1/J**2
        return np.array([[4*p, 2*p,   0],
                         [2*p, 4*p,   0],
                         [0,     0, 2*p]])

class vahid_anisohyper_inv():
    """
    Assumptions:
        Plane stress conditions
        Incompressible material
        SEDF is a function of I1, I2, Iv, Iw, node_X and node_x
    """
    def __init__(self, SEDF):
        self.SEDF = SEDF

    def kinematics(self, F_2D, node_X, node_x):
        C_2D = F_2D.T @ F_2D
        Cinv_2D = np.linalg.inv(C_2D)
        detC_2D = np.linalg.det(C_2D)
        C33 = 1/detC_2D
        C = np.array([[C_2D[0,0], C_2D[0,1], 0],\
                      [C_2D[1,0], C_2D[1,1], 0],\
                      [0,         0,       C33]])
        C2 = C @ C
        Cinv = np.linalg.inv(C)
        I1 = C[0,0] + C[1,1] + C[2,2]
        trC2 = C2[0,0] + C2[1,1] + C2[2,2]
        I2 = 0.5*(I1**2 - trC2)

        theta = self.SEDF.fiberangle(node_X)
        v0 = np.array([ np.cos(theta), np.sin(theta), 0])
        w0 = np.array([-np.sin(theta), np.cos(theta), 0])
        V0 = np.outer(v0, v0)
        W0 = np.outer(w0, w0)
        Iv = np.einsum('ij,ij',C,V0)
        Iw = np.einsum('ij,ij',C,W0)
        return I1, I2, Iv, Iw, C, Cinv, Cinv_2D, V0, W0, C33
    
    def sigma(self, F_2D, node_X=None, node_x=None):
        SEDF = self.SEDF
        I1, I2, Iv, Iw, C, Cinv, _, V0, W0, _ = self.kinematics(F_2D, node_X, node_x)

        Psi1, Psi2, Psiv, Psiw = SEDF.Psi_i(I1, I2, Iv, Iw, node_X)
        
        p = -C[2,2]*(2*Psi1 + 2*Psi2*(I1 - C[2,2]) + 2*Psiv*V0[2,2] + 2*Psiw*W0[2,2])
        S = p*Cinv + 2*Psi1*np.eye(3) + 2*Psi2*(I1*np.eye(3)-C) + 2*Psiv*V0 + 2*Psiw*W0
        S_2D = S[:2,:2]
        return F_2D @ (S_2D @ F_2D.T)
    
    def CC(self, F_2D, node_X, node_x):
        SEDF = self.SEDF
        I1, I2, Iv, Iw, C, _, Cinv_2D, V0, W0, C33 = self.kinematics(F_2D, node_X, node_x)
        C_2D  = C[:2, :2]
        I1_2D = C_2D[0, 0] + C_2D[1, 1]
        V0_2D = V0[:2, :2]
        W0_2D = W0[:2, :2]
        
        # First derivatives of SEDF
        Psi1, Psi2, Psiv, Psiw = SEDF.Psi_i(I1, I2, Iv, Iw, node_X)
        # 2nd derivatives
        Psi11, Psi22, Psivv, Psiww, Psi12, Psi1v, Psi1w, Psi2v, Psi2w, Psivw = SEDF.Psi_ii(I1, I2, Iv, Iw, node_X)

        d1  =  4*(Psi11 + 2*Psi12*(I1_2D + C33) + Psi22*(I1_2D**2 + 2*I1_2D*C33 + C33**2) + Psi2)
        d2  =  4*(-Psi11*C33 - 2*Psi12*I1_2D*C33 - Psi12*C33**2 - Psi22*I1_2D*C33*(I1_2D+C33) - Psi2*C33)
        d3  =  4*(-Psi12 - Psi22*I1_2D - Psi22*C33)
        d4  =  4*(Psi12*C33 + Psi22*I1_2D*C33)
        d5  =  4*Psi22
        d6  = -4*Psi2
        d7  =  4*Psivv
        d8  =  4*Psiww
        d9  =  4*(Psi11*C33**2 + 2*Psi12*I1_2D*C33**2 + Psi1*C33 + Psi22*C33**2*I1_2D**2 + Psi2*I1_2D*C33)
        d10 =  4*(Psi1*C33 + Psi2*I1_2D*C33)
        d11 =  4*(Psi1v + Psi2v*(I1_2D + C33))
        d12 =  4*(Psi1w + Psi2w*(I1_2D + C33))
        d13 =  4*(-Psi1v*C33 - Psi2v*I1_2D*C33)
        d14 =  4*(-Psi1w*C33 - Psi2w*I1*C33)
        d15 = -4*Psi2v
        d16 = -4*Psi2w
        
        I = np.eye(2)
        II = 0.5*(np.einsum('ik,jl->ijkl', I, I) + np.einsum('il,jk->ijkl', I, I))
        dyadic = lambda A, B: np.einsum('ij,kl->ijkl', A, B)
        CC = d1*  dyadic(I,I)                                                                                   + \
             d2* (dyadic(I,Cinv_2D)      + dyadic(Cinv_2D,I)     )                                              + \
             d3* (dyadic(I,C_2D)         + dyadic(C_2D,I)        )                                              + \
             d4* (dyadic(C_2D,Cinv_2D)   + dyadic(Cinv_2D,C_2D)  )                                              + \
             d5*  dyadic(C_2D,C_2D)                                                                             + \
             d6*  II                                                                                            + \
             d7*  dyadic(V0_2D,V0_2D)                                                                           + \
             d8*  dyadic(W0_2D,W0_2D)                                                                           + \
             d9*  dyadic(Cinv_2D, Cinv_2D)                                                                      + \
             d10*0.5*(np.einsum('ik,lj->ijkl',Cinv_2D, Cinv_2D) + np.einsum('il,kj->ijkl', Cinv_2D, Cinv_2D))   + \
             d11*(dyadic(I,V0_2D)       + dyadic(V0_2D,I)       )                                               + \
             d12*(dyadic(I,W0_2D)       + dyadic(W0_2D,I)       )                                               + \
             d13*(dyadic(Cinv_2D,V0_2D) + dyadic(V0_2D,Cinv_2D) )                                               + \
             d14*(dyadic(Cinv_2D,W0_2D) + dyadic(W0_2D,Cinv_2D) )                                               + \
             d15*(dyadic(C_2D,V0_2D)    + dyadic(V0_2D,C_2D)    )                                               + \
             d16*(dyadic(C_2D,W0_2D)    + dyadic(W0_2D,C_2D)    )
        return CC
    
    def D(self, F_2D, node_X=None, node_x=None):
        CC = self.CC(F_2D, node_X, node_x)
        cc = np.einsum('ij,kl,mn,op,jlnp->ikmo', F_2D, F_2D, F_2D, F_2D, CC)

        D = np.zeros((3,3))
        Itoi = [0,1,0]
        Itoj = [0,1,1]
        for I in range(3):
            for J in range(3):
                i = Itoi[I]
                j = Itoj[I]
                k = Itoi[J]
                l = Itoj[J]
                D[I,J] = cc[i,j,k,l]
        return D


def plotmesh(elements, node_X, c, title='mesh', ax=None, fig=None, cmap='Blues', cbar = True, cbar_frac=0.03, extent = None, extend='neither'):
    n_elem = len(elements)

    if ax==None:
        fig, ax = plt.subplots(figsize=(15,7))
    if extent is None:
        extent = [np.min(c), np.max(c)]

    r = np.abs(np.max(node_X)-np.min(node_X))
    ax.set_xlim([np.min(node_X)-0.1*r, np.max(node_X)+0.1*r])
    ax.set_ylim([np.min(node_X)-0.1*r, np.max(node_X)+0.1*r])
    ax.set_aspect('equal')
    patches=[]
    for i in range(n_elem):
        n1 = elements[i,0]
        n2 = elements[i,1]
        n3 = elements[i,2]
        n4 = elements[i,3]
        polygon = Polygon([node_X[n1],node_X[n2],node_X[n3],node_X[n4]], edgecolor='r', facecolor=None, closed=True)
        patches.append(polygon)
    p = PatchCollection(patches, edgecolor='k', facecolor='gray')
    p.set_array(c)
    p.set_cmap(cmap)
    p.set_clim(extent)
    if cbar==True:
        plt.colorbar(p, ax=ax, fraction = cbar_frac, extend=extend)
    ax.add_collection(p)
    ax.set_title(title)
    ax.axis('off') 
    return fig, ax


def write_biax_abaqus_inp(Lambda_params, Lambda_fn, common_params, Ax, Ay, outputfile, disp_or_force='force', inputfile='abaqus/square.inp'):
    # Ax and Ay are the forces in x and y if disp_or_force = 'force' and the displacements of the right and top boundary if disp_or_force = 'disp
    # Assemble the common parameters
    NODE_weights, theta, Psi1_bias, Psi2_bias, alpha = common_params
    params_I1, params_I2, params_1_v, params_1_w, params_v_w = NODE_weights
    params_I1c, params_I1s = params_I1

    aux, unravel = ravel_pytree(params_I1)
    params_Iv   = unravel(aux*0) #Just create dummy empty params lists because the current UANISOHYPER expects all of these
    params_Iw   = unravel(aux*0)
    params_1_2  = unravel(aux*0)
    params_2_v  = unravel(aux*0)
    params_2_w  = unravel(aux*0)

    NODE_weights = [params_I1, params_I2, params_Iv, params_Iw, params_1_2, params_1_v, params_1_w, params_2_v, params_2_w, params_v_w]
    NODE_weights = ravel_pytree(NODE_weights)[0]

    n_input = 1
    n_weights = len(aux)
    n_biases = 11 # not used in Abaqus, so doesn't matter
    n_neurons = len(params_I1c[0][0])
    n_neuronsperlayer = np.array([1, n_neurons, n_neurons, 1])
    n_layers = len(n_neuronsperlayer)
    activtypes = np.array([3, 3, 0])
    Iweights = np.array([0.0, alpha[0], alpha[1], 0.0, 0.0, alpha[2]]) # Just put 0.5 for the unused combinations

    p = np.hstack([n_layers, n_input, n_weights, n_biases, n_neuronsperlayer, NODE_weights, activtypes, Iweights, theta, Psi1_bias, Psi2_bias])
    p = np.pad(p, (0, 8-p.shape[0]%8)) # Increase p by dummy zeros so that it is divisible by 8, just so I have even columns and rows
    nprops = p.shape[0] #376
    p = p.reshape(-1,8)
    commonparams = ',\n'.join(', '.join('%.17f' % num for num in row) for row in p) # This properly converts p into str


    with open(inputfile) as f:
        lines = [line.rstrip('\n') for line in f]

    # Get the nodes
    i1 = -1
    i2 = -1
    for i, line in enumerate(lines): # Find the line that starts with '*Node'
        if line.find('*Node') >= 0:
            i1 = i+1
            break
    for i, line in enumerate(lines): # Find the line that starts with '*Element' 
        if line.find('*Element') >= 0:
            i2 = i
            break
    lines = lines[i1:i2]
    lines = [line.split() for line in lines]
    lines = [[a.rstrip(',') for a in line] for line in lines]
    abq_node_X = np.array(lines, dtype=float)
    nodes = np.array(abq_node_X[:,0],dtype=int)
    abq_node_X = abq_node_X[:,1:3]


    # Write the common parts of the Abaqus input file
    with open(inputfile, 'r') as f:
        inptxt = f.read()

    # Inject the header
    i1 = inptxt.find('*Anisotropic Hyperelastic')
    i2 = i1 + inptxt[i1:].find('\n')
    header = '*Anisotropic Hyperelastic, user, formulation=INVARIANT, type=INCOMPRESSIBLE, local direction=2, properties=' \
        + str(nprops) + ', moduli=INSTANTANEOUS'
    inptxt = inptxt[:i1] + header + inptxt[i2:]

    i = inptxt.find('*Anisotropic Hyperelastic')
    i1 = i + inptxt[i:].find('\n') + 1
    i2 = i + inptxt[i:].find('**')

    inptxt = inptxt[:i1] + commonparams + '\n' + inptxt[i2:]
        
    # Inject the boundary loads
    if disp_or_force=='force':
        i1 = inptxt.find('*Cload')+7
        i2 = i1 + inptxt[i1:].find(',')
        setname = inptxt[i1:i2]
        i2 = i1 + inptxt[i1:].find('*')
        inptxt = inptxt[:i1] + setname + ', 1, ' + str(Ax) + '\n' + inptxt[i2:]

        i1 = i1 + inptxt[i1:].find('*Cload') + 7
        i2 = i1 + inptxt[i1:].find(',')
        setname = inptxt[i1:i2]
        i2 = i1 + inptxt[i1:].find('*')
        inptxt = inptxt[:i1] + setname + ', 2, ' + str(Ay) + '\n' + inptxt[i2:]
    else:
        i1 = inptxt.find('set_rgt_surface, 1, 1')
        i2 = i1 + inptxt[i1:].find('*')
        inptxt = inptxt[:i1] + 'set_rgt_surface, 1, 1, {} \n'.format(Ax) + inptxt[i2:]

        i1 = i1 + inptxt[i1:].find('set_top_surface, 2, 2')
        i2 = i1 + inptxt[i1:].find('*')
        inptxt = inptxt[:i1] + 'set_top_surface, 2, 2, {} \n'.format(Ay) + inptxt[i2:]



    # Write the heterogeneous parts of the Abaqus input file
    # Find the instance name
    i1 = inptxt.find('*Instance')
    i1 = i1 + inptxt[i1:].find('=') + 1
    i2 = i1 + inptxt[i1:].find(',')
    instname = inptxt[i1:i2]

    # Define node sets for each node
    i = inptxt.find('*Assembly')
    i = i + inptxt[i:].find('*Nset')
    part1, part2 = inptxt[:i], inptxt[i:]
    for node in nodes:
        part1 += "*Nset, nset=myset-" + str(node) + ", instance=" + instname + " \n " + str(node) + ",\n"
    inptxt = part1+part2

    # Define field variables for each node set
    i = inptxt.find("** ----")
    part1, part2 = inptxt[:i], inptxt[i:]
    part1 += "** \n** PREDEFINED FIELDS\n** \n"
    k = 1
    for i in range(len(nodes)):
        node = nodes[i]
        Lambda_inp = abq_node_X[i].reshape([-1,2])
        phi = Lambda_fn(Lambda_inp, Lambda_params).flatten()

        for j, param in enumerate(phi):
            part1 += "** Name: myfield-" + str(k) + "   Type: Field\n*Initial Conditions, type=FIELD, variable=" + \
                str(j+1) + "\nmyset-" + str(node) + ", " + str(np.exp(param)) + "\n"
            k+= 1
    inptxt = part1+part2

    with open(outputfile, 'w') as f:
        f.write(inptxt)



def apply_bc_biax(node_X, lmx, lmy, l0=1.0, h0=1.0):
    n_node = len(node_X)
    node_x = np.zeros_like(node_X)
    DOF_fmap = np.zeros((n_node,2),dtype=int)
    dof = 0
    for i in range(n_node):
        X = node_X[i]
        node_x[i] = node_X[i]
        if X[0]<0.001 and X[1]<0.001: #lower left corner
            DOF_fmap[i,0] = -1
            DOF_fmap[i,1] = -1
            node_x[i,0] = 0
            node_x[i,1] = 0
        elif X[0]>0.999*l0 and X[1]<0.001: #lower right corner
            DOF_fmap[i,0] = -1
            DOF_fmap[i,1] = -1
            node_x[i,0] = lmx
            node_x[i,1] = 0
        elif X[0]>0.999*l0 and X[1]>0.999*h0: #upper right corner
            DOF_fmap[i,0] = -1
            DOF_fmap[i,1] = -1
            node_x[i,0] = lmx
            node_x[i,1] = lmy
        elif X[0]<0.001 and X[1]>0.999*h0: #upper left corner
            DOF_fmap[i,0] = -1
            DOF_fmap[i,1] = -1
            node_x[i,0] = 0
            node_x[i,1] = lmy
        elif X[0]<0.001: #The rest of the left boundary
            DOF_fmap[i,0] = -1
            node_x[i,0] = 0
            DOF_fmap[i,1] = dof
            dof+= 1
        elif X[1]<0.001: #The rest of the lower boundary
            DOF_fmap[i,1] = -1
            node_x[i,1] = 0
            DOF_fmap[i,0] = dof
            dof+= 1
        elif X[0]>0.999*l0: #The rest of the right boundary
            DOF_fmap[i,0] = -1
            node_x[i,0] = lmx
            DOF_fmap[i,1] = dof
            dof+= 1
        elif X[1]>0.999*h0: #The rest of the upper boundary
            DOF_fmap[i,1] = -1
            node_x[i,1] = lmy
            DOF_fmap[i,0] = dof
            dof+= 1
        else: #The rest of the domain
            DOF_fmap[i,0] = dof
            dof+=1
            DOF_fmap[i,1] = dof
            dof+=1
    return node_x, DOF_fmap
    
def apply_bc_uniax_x(node_X, lmx):
    n_node = len(node_X)
    node_x = np.zeros_like(node_X)
    DOF_fmap = np.zeros((n_node,2),dtype=int)
    dof = 0
    for i in range(n_node):
        X = node_X[i]
        node_x[i] = node_X[i]
        if X[0]<0.001:
            DOF_fmap[i,0] = -1
            node_x[i,0] = 0
            if X[1] < 0.001:
                DOF_fmap[i,1] = -1
                node_x[i,1] = 0
            else:
                DOF_fmap[i,1] = dof
                dof+= 1
        elif X[0]>0.999:
            DOF_fmap[i,0] = -1
            node_x[i,0] = lmx
            DOF_fmap[i,1] = dof
            dof+= 1
        else:
            DOF_fmap[i,0] = dof
            DOF_fmap[i,1] = dof+1
            dof+=2 
    return node_x, DOF_fmap