import os
n_cores = 4
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count={}'.format(n_cores)

from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import pickle
from jax import jit, vmap, grad, jacrev
from jax.lax import cond, scan
import jax.random as random
key = random.PRNGKey(2022)
import jax.example_libraries.optimizers as optimizers
from jax.flatten_util import ravel_pytree
from jax.sharding import PositionalSharding
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd

from utils import train, coords_2_strain_nn, nn_fpass
from utils import train_colloc_parallel as train_colloc, init_params_nn, ff_nn, divergence, bd_forces, a1, a2, lr
from utils_hyperelasticity import NODE, init_layers, NODE_model_iso, init_params_iso, GOH_model, eval_Cauchy, eval_Cauchy_vmap, eval_P, eval_P_vmap
from utils_hyperelasticity import ThreeDElasticity, merge_c_s_params_iso, split_c_s_params_iso
from fem import plotmesh, plotmesh_updated, fe_solver_2D, vahid_anisohyper_inv, apply_bc_biax

from jax_fem.core import FEM
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh, get_meshio_cell_type, Mesh, rectangle_mesh

from PIL import Image, ImageFilter
from jaxinterp2d import interp2d

E_fun = vmap(lambda F: 0.5*(F.T@F - jnp.eye(2)))


with open('/home/amir/inverse_prob/params_for_noise_tuning/manu/image_93_info.npy', 'rb') as f:
    _,_,_,P_xx_gt,P_yy_gt = pickle.load(f)
with open('/home/amir/inverse_prob/params_for_noise_tuning/manu/image93_disp.npy', 'rb') as f:
    x,y,ux_hist,uy_hist = pickle.load(f)
with open('/home/amir/inverse_prob/params_for_noise_tuning/manu/_image93_temp.npy', 'rb') as f:
    E_hist_dic = pickle.load(f)
uy_hist=-uy_hist 

x1 = 4
x2 = 50
y1 = 2
y2 = 48
out = []
for aux in E_hist_dic:
    out.append(aux.transpose([2,0,1]).reshape([51,51,2,2]).transpose([1,0,2,3])[x1:x2,y1:y2,:,:].reshape([-1,2,2])) # I know. wtf?
E_hist_dic = np.array(out)
elem_X = np.stack([x.flatten(),y.flatten()]).T


# we don't need to use all of that data
i1 = 10
i2 = None
skip = 40
P_xx_gt = jnp.array(P_xx_gt[i1:i2:skip])
P_yy_gt = jnp.array(P_yy_gt[i1:i2:skip])
ux_hist = jnp.array(ux_hist[i1:i2:skip])
uy_hist = jnp.array(uy_hist[i1:i2:skip])

E_hist_dic = E_hist_dic[i1:i2:skip]
#%%

ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)

Nx, Ny = x.shape[0], x.shape[1]
Lx, Ly = 1.0, 1.0
meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])
range_minmax=np.zeros((2,2))
disps_norm = []
n_panels = 5
for i in range(len(ux_hist)):
    if i==0:
        range_minmax[0,0]=np.min(ux_hist[i])
        range_minmax[0,1]=np.max(ux_hist[i])
        range_minmax[1,0]=np.min(uy_hist[i])
        range_minmax[1,1]=np.max(uy_hist[i])
    else:
        range_minmax[0,0]=np.min(np.array([range_minmax[0,0],np.min(ux_hist[i])]))
        range_minmax[0,1]=np.max(np.array([range_minmax[0,1],np.max(ux_hist[i])]))
        range_minmax[1,0]=np.min(np.array([range_minmax[1,0],np.min(uy_hist[i])]))
        range_minmax[1,1]=np.max(np.array([range_minmax[1,1],np.max(uy_hist[i])]))

#%%
ele_type = 'QUAD4'
cell_type = get_meshio_cell_type(ele_type)

Nx, Ny = x.shape[0], x.shape[1]
Lx, Ly = 1.0, 1.0
meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

disps_norm = []
n_panels = 5
fig, axes = plt.subplots(2,n_panels,figsize=(5*n_panels,8))
t_hist = np.arange(len(ux_hist))
disps_x_mean = []
disps_x_stdv = []
disps_y_mean = []
disps_y_stdv = []
for i in range(len(ux_hist)):
    t = t_hist[i]
    plotmesh_updated(mesh.cells, mesh.points, ux_hist[i].flatten(), \
                     range_minmax[0,0],range_minmax[0,1],title='Displacement in x', ax=axes[0,i], ec='None'); 
    plotmesh_updated(mesh.cells, mesh.points, uy_hist[i].flatten(), \
                     range_minmax[1,0],range_minmax[1,1],title='Displacement in y', ax=axes[1,i], ec='None'); 

    temp1 = ux_hist[i].flatten()
    temp2 = uy_hist[i].flatten()
    disps_x_mean.append(temp1.mean())
    disps_x_stdv.append(temp1.std())
    disps_y_mean.append(temp2.mean())
    disps_y_stdv.append(temp2.std())

    temp1 = (temp1 - temp1.mean())/temp1.std()
    temp2 = (temp2 - temp2.mean())/temp2.std()

    disps_norm.append(np.array([t*np.ones(len(temp1)), temp1, temp2]))
disps_norm = np.hstack(disps_norm)
disps_x_mean = jnp.hstack(disps_x_mean)
disps_x_stdv = jnp.hstack(disps_x_stdv)
disps_y_mean = jnp.hstack(disps_y_mean)
disps_y_stdv = jnp.hstack(disps_y_stdv)
#%%
ux_hist=np.array(ux_hist)
uy_hist=np.array(uy_hist)
X_grid=elem_X[:,0].reshape((Nx,Ny)).T
Y_grid=elem_X[:,1].reshape((Nx,Ny)).T[::-1,:]
u_hist_for_derivative=[]
for ii in range(ux_hist.shape[0]):
        ux_hist
        u_hist_for_derivative.append([X_grid,Y_grid,ux_hist[ii,:,:],uy_hist[ii,:,:]])
#%%
def richardson_derivative(Z, h):
    """Calculates the derivatives using finite difference and Richardson extrapolation for step size h."""
    Z_x = np.zeros_like(Z)
    Z_y = np.zeros_like(Z)
    rows, cols = Z.shape
    for i in range(rows):
        for j in range(cols):
            if i == 0:  
                Z_y[i, j] = (Z[i + 1, j] - Z[i, j]) / h
            elif i == rows - 1:  
                Z_y[i, j] = (Z[i, j] - Z[i - 1, j]) / h
            else:  
                dy_h = (Z[i + 1, j] - Z[i - 1, j]) / (2 * h)
                
        
                f_y_plus_h_half = (Z[i + 1, j] + Z[i, j]) / 2
                f_y_minus_h_half = (Z[i - 1, j] + Z[i, j]) / 2
                
                dy_h_over_2 = (f_y_plus_h_half - f_y_minus_h_half) / h

               
                #Z_y[i, j] = (4 * dy_h_over_2 - dy_h) / 3
                Z_y[i, j] = dy_h
            if j == 0: 
                Z_x[i, j] = (Z[i, j + 1] - Z[i, j]) / h
            elif j == cols - 1: 
                Z_x[i, j] = (Z[i, j] - Z[i, j - 1]) / h
            else:  
                dx_h = (Z[i, j + 1] - Z[i, j - 1]) / (2 * h)
                
                f_x_plus_h_half = (Z[i, j + 1] + Z[i, j]) / 2
                f_x_minus_h_half = (Z[i, j - 1] + Z[i, j]) / 2
                
                dx_h_over_2 = (f_x_plus_h_half - f_x_minus_h_half) / h

                #Z_x[i, j] = (4 * dx_h_over_2 - dx_h) / 3
                Z_x[i, j] = dx_h

    return Z_x,Z_y
F_with_FDM=np.zeros((5,46*46,2,2))
for ii in range(len(u_hist_for_derivative)):
           a,b,ux_grid,uy_grid=u_hist_for_derivative[ii]
           #print(ux_grid)
           uxx,uxy=richardson_derivative(ux_grid.T,1/(Nx-1))
           #print('derivative')
           #print(uxx)
           uyx,uyy=richardson_derivative(uy_grid.T,1/(Nx-1))
           id_mesh=0
           for i in range(uxx.shape[1]):
                for j in range(uxx.shape[0]):
                     F_with_FDM[ii,id_mesh, :,:]=np.array([[uxx[j,i]+1,uxy[j,i]],[uyx[j,i],uyy[j,i]+1]])
                     id_mesh+=1
#print(uy_grid.T[::-1,:])
# plt.scatter(elem_X[:,0], elem_X[:,1])
# a=elem_X[:,0].reshape((Nx,Ny)).T
# b=elem_X[:,1].reshape((Nx,Ny)).T[::-1,:]
# print(a)
# print(b)
