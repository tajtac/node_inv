# %%
import os
n_cores = 20
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count={}'.format(n_cores)

from jax.config import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import pickle
from jax import jit, vmap, grad
import jax.random as random
key = random.PRNGKey(2022)
import jax.example_libraries.optimizers as optimizers
from jax.flatten_util import ravel_pytree
from jax.sharding import PositionalSharding
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd

from utils import train, eval_Cauchy_aniso, eval_Cauchy_aniso_vmap, merge_weights_aniso
from utils_hyperelasticity import NODE, init_layers, NODE_model_aniso, init_params_aniso
from fem import plotmesh, fe_solver_2D, vahid_anisohyper_inv, write_biax_abaqus_inp

from jaxinterp2d import interp2d

import time

# %%
def nn_fpass(H, params):
    Ws, bs = params
    N_layers = len(Ws)
    for i in range(N_layers - 1):
        H = jnp.matmul(H, Ws[i]) + bs[i]
        H = jnp.tanh(H)
    H = jnp.matmul(H, Ws[-1]) + bs[-1]
    return H

def ff_nn(x, params): #ff: fourier features
    ff_params, nn_params = params
    x = jnp.matmul(x, ff_params)
    x = jnp.hstack([jnp.sin(2*jnp.pi*x), jnp.cos(2*jnp.pi*x)])

    x = nn_fpass(x, nn_params)
    return x

def init_params_nn(layers, key):
  Ws = []
  bs = []
  for i in range(len(layers) - 1):
    std_glorot = jnp.sqrt(2/(layers[i] + layers[i + 1]))
    key, subkey = random.split(key)
    Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
    bs.append(jnp.zeros(layers[i + 1]))
  return [Ws, bs]



# %%
with open('params/circ_s5_pre.npy', 'rb') as f:
    coord_2_strain_params, node_params, Lambda_params, node_X, elements, n_node, n_elem, Fx, Fy, strains, \
        rgt_bd_X, top_bd_X, lft_bd_X, bot_bd_X = pickle.load(f)
_, unravel = ravel_pytree(node_params)



# %%
# Train with the heterogeneity
# ff_params = coord_2_strain_params[0]
# ff_nn2 = lambda x, nn_params: ff_nn(x, [ff_params, nn_params])
def get_stresses(x, y, Lambda_params):
    epsx, epsy = ff_nn(jnp.array([x,y]), coord_2_strain_params).T

    # get NODE individual-specific params, phi, from the Lambda NN
    Lambda_inp = jnp.array([x,y]).reshape([-1,2])
    phi = ff_nn(Lambda_inp, Lambda_params).flatten()
    # Make predictions with this NODE
    mymodel = NODE_model_aniso(unravel(phi))
    sgm = eval_Cauchy_aniso(epsx+1.0,epsy+1.0, mymodel)
    return sgm[0,0], sgm[1,0], sgm[0,1], sgm[1,1]
get_stresses_vmap = vmap(get_stresses, in_axes=(0,0,None), out_axes=(0,0,0,0))

get_sgm_xx = lambda x, y, Lambda_params: get_stresses(x, y, Lambda_params)[0]
get_sgm_yx = lambda x, y, Lambda_params: get_stresses(x, y, Lambda_params)[1]
get_sgm_xy = lambda x, y, Lambda_params: get_stresses(x, y, Lambda_params)[2]
get_sgm_yy = lambda x, y, Lambda_params: get_stresses(x, y, Lambda_params)[3]

grad_sgm_xx_x = vmap(grad(get_sgm_xx, argnums=0), in_axes=(0,0,None), out_axes=0) # should return dsgm_xx/dx
grad_sgm_yx_x = vmap(grad(get_sgm_yx, argnums=0), in_axes=(0,0,None), out_axes=0)
grad_sgm_xy_y = vmap(grad(get_sgm_yx, argnums=1), in_axes=(0,0,None), out_axes=0)
grad_sgm_yy_y = vmap(grad(get_sgm_yy, argnums=1), in_axes=(0,0,None), out_axes=0)

@jit
def bd_forces(Lambda_params, input):
    rgt_bd_sgm = get_stresses_vmap(rgt_bd_X[:,0], rgt_bd_X[:,1], Lambda_params)
    top_bd_sgm = get_stresses_vmap(top_bd_X[:,0], top_bd_X[:,1], Lambda_params)
    lft_bd_sgm = get_stresses_vmap(lft_bd_X[:,0], lft_bd_X[:,1], Lambda_params)
    bot_bd_sgm = get_stresses_vmap(bot_bd_X[:,0], bot_bd_X[:,1], Lambda_params)

    rgt_bd_frc = jnp.sum(rgt_bd_sgm[0]) # Assume area of edge = 1.0
    top_bd_frc = jnp.sum(top_bd_sgm[3])
    lft_bd_frc = jnp.sum(lft_bd_sgm[0])
    bot_bd_frc = jnp.sum(bot_bd_sgm[3])
    return rgt_bd_frc, top_bd_frc, lft_bd_frc, bot_bd_frc
@jit
def divergence(Lambda_params, X_colloc):
    # X_colloc = random.uniform(key, (500,2))
    dsgm_xx_dx = grad_sgm_xx_x(X_colloc[:,0], X_colloc[:,1], Lambda_params)
    dsgm_xy_dy = grad_sgm_xy_y(X_colloc[:,0], X_colloc[:,1], Lambda_params)
    dsgm_yx_dx = grad_sgm_yx_x(X_colloc[:,0], X_colloc[:,1], Lambda_params)
    dsgm_yy_dy = grad_sgm_yy_y(X_colloc[:,0], X_colloc[:,1], Lambda_params)
    
    div_x = jnp.mean(dsgm_xx_dx + dsgm_xy_dy) # = dsgm_xx/dx + dsgm_xy/dy
    div_y = jnp.mean(dsgm_yx_dx + dsgm_yy_dy) # = dsgm_yx/dx + dsgm_yy/dy
    return div_x, div_y
@jit
def loss(Lambda_params, X_colloc):
    div_x, div_y = divergence(Lambda_params, X_colloc)
    rgt_bd_frc, top_bd_frc, lft_bd_frc, bot_bd_frc = bd_forces(Lambda_params, None)
    a1 = 500.0
    a2 = 1.0
    return a1*(div_x**2 + div_y**2) + a2*((rgt_bd_frc-Fx)**2 + (top_bd_frc-Fy)**2 + (lft_bd_frc-Fx)**2 + (bot_bd_frc-Fy)**2)

@partial(jit, static_argnums=(0,2,3,))
def step(loss, i, get_params, opt_update, opt_state, X_batch):
    params = get_params(opt_state)
    g = grad(loss)(params, X_batch)
    return opt_update(i, g, opt_state)

def train(loss, X, get_params, opt_update, opt_state, key, nIter = 10000, print_freq=1000, metric_fns=None, batch_size=500):
    val_loss = []
    metrics = []
    for it in range(nIter):
        key, subkey = random.split(key)
        X_colloc = random.uniform(key, (batch_size,2))
        X_colloc = jax.device_put(X_colloc, sharding)
        opt_state = step(loss, it, get_params, opt_update, opt_state, X_colloc)
        if (it+1)% print_freq == 0:
            params = get_params(opt_state)
            val_loss_value = loss(params, X)
            val_loss.append(val_loss_value)
            if metric_fns is not None:
                m = []
                for metric_fn in metric_fns:
                    m.append(metric_fn(params, X))
                metrics.append(m)
    return get_params(opt_state), val_loss, metrics

sharding = PositionalSharding(jax.devices()).reshape(n_cores, 1)
opt_init, opt_update, get_params = optimizers.adam(5.e-4) #Original: 5.e-4
opt_state = opt_init(Lambda_params)


#Add some training to the beginning so that the functions are jitted
Lambda_params, val_loss, metrics = train(loss, node_X, get_params, opt_update, opt_state, 
                                              key, nIter = 1_000, print_freq=1000, metric_fns=[bd_forces, divergence], batch_size=100)


# print('Training with batch_size=100...')
# t1 = time.time()
# Lambda_params, val_loss, metrics = train(loss, node_X, get_params, opt_update, opt_state, 
#                                               key, nIter = 1_000, print_freq=1000, metric_fns=[bd_forces, divergence], batch_size=100)
# t2 = time.time()
# print('Done. Elapsed time = {} min'.format((t2-t1)/60.0))


# print('Training with batch_size=200...')
# t1 = time.time()
# Lambda_params, val_loss, metrics = train(loss, node_X, get_params, opt_update, opt_state, 
#                                               key, nIter = 1_000, print_freq=1000, metric_fns=[bd_forces, divergence], batch_size=200)
# t2 = time.time()
# print('Done. Elapsed time = {} min'.format((t2-t1)/60.0))


# print('Training with batch_size=400...')
# t1 = time.time()
# Lambda_params, val_loss, metrics = train(loss, node_X, get_params, opt_update, opt_state, 
#                                               key, nIter = 1_000, print_freq=1000, metric_fns=[bd_forces, divergence], batch_size=400)
# t2 = time.time()
# print('Done. Elapsed time = {} min'.format((t2-t1)/60.0))


# print('Training with batch_size=800...')
# t1 = time.time()
# Lambda_params, val_loss, metrics = train(loss, node_X, get_params, opt_update, opt_state, 
#                                               key, nIter = 1_000, print_freq=1000, metric_fns=[bd_forces, divergence], batch_size=800)
# t2 = time.time()
# print('Done. Elapsed time = {} min'.format((t2-t1)/60.0))


# print('Training with batch_size=1600...')
# t1 = time.time()
# Lambda_params, val_loss, metrics = train(loss, node_X, get_params, opt_update, opt_state, 
#                                               key, nIter = 1_000, print_freq=1000, metric_fns=[bd_forces, divergence], batch_size=1600)
# t2 = time.time()
# print('Done. Elapsed time = {} min'.format((t2-t1)/60.0))


# print('Training with batch_size=3200...')
# t1 = time.time()
# Lambda_params, val_loss, metrics = train(loss, node_X, get_params, opt_update, opt_state, 
#                                               key, nIter = 1_000, print_freq=1000, metric_fns=[bd_forces, divergence], batch_size=3200)
# t2 = time.time()
# print('Done. Elapsed time = {} min'.format((t2-t1)/60.0))


# print('Training with batch_size=6400...')
# t1 = time.time()
# Lambda_params, val_loss, metrics = train(loss, node_X, get_params, opt_update, opt_state, 
#                                               key, nIter = 1_000, print_freq=1000, metric_fns=[bd_forces, divergence], batch_size=6400)
# t2 = time.time()
# print('Done. Elapsed time = {} min'.format((t2-t1)/60.0))


print('Training with batch_size=12800...')
t1 = time.time()
Lambda_params, val_loss, metrics = train(loss, node_X, get_params, opt_update, opt_state, 
                                              key, nIter = 1_000, print_freq=1000, metric_fns=[bd_forces, divergence], batch_size=12800)
t2 = time.time()
print('Done. Elapsed time = {} min'.format((t2-t1)/60.0))


print('Training with batch_size=25600...')
t1 = time.time()
Lambda_params, val_loss, metrics = train(loss, node_X, get_params, opt_update, opt_state, 
                                              key, nIter = 1_000, print_freq=1000, metric_fns=[bd_forces, divergence], batch_size=25600)
t2 = time.time()
print('Done. Elapsed time = {} min'.format((t2-t1)/60.0))


print('Training with batch_size=51200...')
t1 = time.time()
Lambda_params, val_loss, metrics = train(loss, node_X, get_params, opt_update, opt_state, 
                                              key, nIter = 1_000, print_freq=1000, metric_fns=[bd_forces, divergence], batch_size=51200)
t2 = time.time()
print('Done. Elapsed time = {} min'.format((t2-t1)/60.0))


print('Training with batch_size=102400...')
t1 = time.time()
Lambda_params, val_loss, metrics = train(loss, node_X, get_params, opt_update, opt_state, 
                                              key, nIter = 1_000, print_freq=1000, metric_fns=[bd_forces, divergence], batch_size=102400)
t2 = time.time()
print('Done. Elapsed time = {} min'.format((t2-t1)/60.0))


print('Training with batch_size=204800...')
t1 = time.time()
Lambda_params, val_loss, metrics = train(loss, node_X, get_params, opt_update, opt_state, 
                                              key, nIter = 1_000, print_freq=1000, metric_fns=[bd_forces, divergence], batch_size=204800)
t2 = time.time()
print('Done. Elapsed time = {} min'.format((t2-t1)/60.0))


print('Training with batch_size=409600...')
t1 = time.time()
Lambda_params, val_loss, metrics = train(loss, node_X, get_params, opt_update, opt_state, 
                                              key, nIter = 1_000, print_freq=1000, metric_fns=[bd_forces, divergence], batch_size=409600)
t2 = time.time()
print('Done. Elapsed time = {} min'.format((t2-t1)/60.0))

