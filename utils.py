import jax.numpy as jnp
import numpy as np
import jax
from jax import grad, vmap, jit, jacrev
from functools import partial
import jax.random as random
#from jax.experimental import optimizers
import jax.example_libraries.optimizers as optimizers
from jax.scipy.optimize import minimize
from jax.lax import scan
from jax.nn import softplus
from jax.config import config
from jax.flatten_util import ravel_pytree

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import pickle

config.update("jax_enable_x64", True)

def eval_Cauchy(lmbx,lmby, model):
    lmbz = 1.0/(lmbx*lmby)
    F = jnp.array([[lmbx, 0, 0],
                   [0, lmby, 0],
                   [0, 0, lmbz]])
    C = F.T @ F
    C2 = C @ C
    Cinv = jnp.linalg.inv(C)
    I1 = C[0,0] + C[1,1] + C[2,2]
    trC2 = C2[0,0] + C2[1,1] + C2[2,2]
    I2 = 0.5*(I1**2 - trC2)

    Psi1 = model.Psi1(I1, I2)
    Psi2 = model.Psi2(I1, I2)
    
    p = -C[2,2]*(2*Psi1 + 2*Psi2*(I1 - C[2,2]))
    S = p*Cinv + 2*Psi1*jnp.eye(3) + 2*Psi2*(I1*jnp.eye(3)-C)
    sgm = F @ (S @ F.T)
    return sgm
eval_Cauchy_vmap = vmap(eval_Cauchy, in_axes=(0,0,None), out_axes = 0)

def eval_Cauchy_aniso(lmbx,lmby, model):
    lmbz = 1.0/(lmbx*lmby)
    F = jnp.array([[lmbx, 0, 0],
                   [0, lmby, 0],
                   [0, 0, lmbz]])
    C = F.T @ F
    C2 = C @ C
    Cinv = jnp.linalg.inv(C)
    theta = model.theta
    v0 = jnp.array([ jnp.cos(theta), jnp.sin(theta), 0])
    w0 = jnp.array([-jnp.sin(theta), jnp.cos(theta), 0])
    V0 = jnp.outer(v0, v0)
    W0 = jnp.outer(w0, w0)

    I1 = C[0,0] + C[1,1] + C[2,2]
    trC2 = C2[0,0] + C2[1,1] + C2[2,2]
    I2 = 0.5*(I1**2 - trC2)
    Iv = jnp.einsum('ij,ij',C,V0)
    Iw = jnp.einsum('ij,ij',C,W0)

    Psi1 = model.Psi1(I1, I2, Iv, Iw)
    Psi2 = model.Psi2(I1, I2, Iv, Iw)
    Psiv = model.Psiv(I1, I2, Iv, Iw)
    Psiw = model.Psiw(I1, I2, Iv, Iw)

    p = -C[2,2]*(2*Psi1 + 2*Psi2*(I1 - C[2,2]) + 2*Psiv*V0[2,2] + 2*Psiw*W0[2,2])
    S = p*Cinv + 2*Psi1*jnp.eye(3) + 2*Psi2*(I1*jnp.eye(3)-C) + 2*Psiv*V0 + 2*Psiw*W0

    sgm = F @ (S @ F.T)
    return sgm
eval_Cauchy_aniso_vmap = vmap(eval_Cauchy_aniso, in_axes=(0,0,None), out_axes = 0)

def merge_weights_aniso(params_c, params_s):
    NODE_weights, theta, Psi1_bias, Psi2_bias, alpha = params_c
    params_I1, params_I2, params_1_v, params_1_w, params_v_w = NODE_weights
    params_I1c,params_I1s = params_I1
    params_I2c,params_I2s = params_I2
    params_1_vc,params_1_vs = params_1_v
    params_1_wc,params_1_ws = params_1_w
    params_v_wc,params_v_ws = params_v_w

    params_I1   = (params_I1c,  params_s[0])
    params_I2   = (params_I2c,  params_s[1])
    params_1_v  = (params_1_vc, params_s[2])
    params_1_w  = (params_1_wc, params_s[3])
    params_v_w  = (params_v_wc, params_s[4])
    NODE_weights = (params_I1, params_I2, params_1_v, params_1_w, params_v_w)
    params = [NODE_weights, params_s[5], params_s[6], params_s[7], params_s[8]]
    return params



"""
    Unified training functions
"""
@partial(jit, static_argnums=(0,2,3,))
def step(loss, i, get_params, opt_update, opt_state, X_batch, key):
    params = get_params(opt_state)
    g = grad(loss)(params, X_batch, key)
    return opt_update(i, g, opt_state)

def train(loss, X, get_params, opt_update, opt_state, key, nIter = 10000, print_freq=1000, metric_fns=None, batch_size=None):
    train_loss = []
    metrics = []
    for it in range(nIter):
        key, subkey = random.split(key)
        if batch_size is None:
            X_batch = X
        else:
            # X_batch = random.choice(key=key, a=X, shape=(batch_size,), replace=False) # This is too slow when X is big
            idx = random.randint(key, minval=0, maxval=len(X), shape=(batch_size,)) #This can have repeated values, but that is ok
            X_batch = X[idx]
        opt_state = step(loss, it, get_params, opt_update, opt_state, X_batch, key)         
        if (it+1)% print_freq == 0:
            params = get_params(opt_state)
            train_loss_value = loss(params, X_batch, key)
            train_loss.append(train_loss_value)
            if metric_fns is not None:
                m = []
                for metric_fn in metric_fns:
                    m.append(metric_fn(params, X, key))
                metrics.append(m)
            to_print = "it %i, train loss = %e" % (it+1, train_loss_value)
            print(to_print)
    return get_params(opt_state), train_loss, metrics


@partial(jit, static_argnums=(0,2,3,))
def step_colloc(loss, i, get_params, opt_update, opt_state, X_batch):
    params = get_params(opt_state)
    g = grad(loss)(params, X_batch)
    return opt_update(i, g, opt_state)

def train_colloc(loss, X, get_params, opt_update, opt_state, key, fname, nIter = 10000, print_freq=1000, metric_fns=None):
    val_loss = []
    metrics = []
    for it in range(nIter):
        key, subkey = random.split(key)
        X_colloc = random.uniform(key, (10000,2))
        opt_state = step_colloc(loss, it, get_params, opt_update, opt_state, X_colloc)
        if (it+1)% print_freq == 0:
            params = get_params(opt_state)
            with open('params/incr/'+fname+'_epoch_{}.npy'.format(it+1), 'wb') as f:
                pickle.dump(params, f)
            val_loss_value = loss(params, X)
            val_loss.append(val_loss_value)
            if metric_fns is not None:
                m = []
                for metric_fn in metric_fns:
                    m.append(metric_fn(params, X))
                metrics.append(m)
            to_print = "it {}, val loss = {:e}".format(it+1, val_loss_value)
            print(to_print)
    return get_params(opt_state), val_loss, metrics

def train_colloc_parallel(loss, inp, get_params, opt_update, opt_state, key, sharding, fname, nIter = 10000, print_freq=1000, metric_fns=None):
    node_X, t_hist = inp
    node_X_ext = jnp.hstack([node_X, t_hist[-1]*np.ones_like(node_X[:,:1])]) #append a row consisting of t_hist[-1]
    val_loss = []
    metrics = []
    for it in range(nIter):
        key, subkey = random.split(key)
        X_colloc = random.uniform(key, (10000,2))
        t_colloc = random.choice(key, t_hist, (10000,1))
        X_colloc = jnp.hstack([X_colloc, t_colloc])
        X_colloc = jax.device_put(X_colloc, sharding)
        X = [X_colloc, random.choice(key, t_hist)]
        opt_state = step_colloc(loss, it, get_params, opt_update, opt_state, X)
        if (it+1)% print_freq == 0:
            params = get_params(opt_state)
            # with open('params/incr/'+fname+'_epoch_{}.npy'.format(it+1), 'wb') as f:
            #     pickle.dump(params, f)
            val_loss_value = loss(params, [node_X_ext, t_hist[-1]])
            val_loss.append(val_loss_value)
            if metric_fns is not None:
                m = []
                for metric_fn in metric_fns:
                    m.append(metric_fn(params, [node_X_ext, t_hist[-1]]))
                metrics.append(m)
            to_print = "it {}, val loss = {:e}".format(it+1, val_loss_value)
            print(to_print)
    return get_params(opt_state), val_loss, metrics

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

def coords_2_strain_nn(x, params):
    ff_params, nn_params = params
    x, t = x[:,:2], x[:,2:3]
    x = jnp.matmul(x, ff_params)
    x = jnp.hstack([jnp.sin(2*jnp.pi*x), jnp.cos(2*jnp.pi*x), t])

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

def get_stresses(x, y, t, Lambda_params, coord_2_strain_params, model):
    epsx, epsy = coords_2_strain_nn(jnp.array([x,y,t])[None,:], coord_2_strain_params).flatten()

    # get NODE individual-specific params, phi, from the Lambda NN
    Lambda_inp = jnp.array([x,y]).reshape([-1,2])
    phi = ff_nn(Lambda_inp, Lambda_params).flatten()
    # Make predictions with this NODE
    mymodel = model(phi)
    sgm = eval_Cauchy_aniso(epsx+1.0,epsy+1.0, mymodel)
    return sgm[0,0], sgm[1,0], sgm[0,1], sgm[1,1]
get_stresses_vmap = vmap(get_stresses, in_axes=(0,0,0,None,None,None), out_axes=(0,0,0,0))

get_sgm_xx = lambda x, y, t, a, b, c: get_stresses(x, y, t, a, b, c)[0]
get_sgm_yx = lambda x, y, t, a, b, c: get_stresses(x, y, t, a, b, c)[1]
get_sgm_xy = lambda x, y, t, a, b, c: get_stresses(x, y, t, a, b, c)[2]
get_sgm_yy = lambda x, y, t, a, b, c: get_stresses(x, y, t, a, b, c)[3]

grad_sgm_xx_x = vmap(grad(get_sgm_xx, argnums=0), in_axes=(0,0,0,None,None,None), out_axes=0) # should return dsgm_xx/dx
grad_sgm_yx_x = vmap(grad(get_sgm_yx, argnums=0), in_axes=(0,0,0,None,None,None), out_axes=0)
grad_sgm_xy_y = vmap(grad(get_sgm_yx, argnums=1), in_axes=(0,0,0,None,None,None), out_axes=0)
grad_sgm_yy_y = vmap(grad(get_sgm_yy, argnums=1), in_axes=(0,0,0,None,None,None), out_axes=0)

@partial(jit, static_argnums=(4,))
def bd_forces(Lambda_params, X, bd_X, c2s_params, model):
    rgt_bd_X, top_bd_X, lft_bd_X, bot_bd_X = bd_X
    t = X[1]
    rgt_bd_sgm = get_stresses_vmap(rgt_bd_X[:,0], rgt_bd_X[:,1], t*jnp.ones_like(rgt_bd_X[:,0]), Lambda_params, c2s_params, model)
    top_bd_sgm = get_stresses_vmap(top_bd_X[:,0], top_bd_X[:,1], t*jnp.ones_like(top_bd_X[:,0]), Lambda_params, c2s_params, model)
    lft_bd_sgm = get_stresses_vmap(lft_bd_X[:,0], lft_bd_X[:,1], t*jnp.ones_like(lft_bd_X[:,0]), Lambda_params, c2s_params, model)
    bot_bd_sgm = get_stresses_vmap(bot_bd_X[:,0], bot_bd_X[:,1], t*jnp.ones_like(bot_bd_X[:,0]), Lambda_params, c2s_params, model)

    rgt_bd_frc = jnp.sum(rgt_bd_sgm[0]) # Assume area of edge = 1.0
    top_bd_frc = jnp.sum(top_bd_sgm[3])
    lft_bd_frc = jnp.sum(lft_bd_sgm[0])
    bot_bd_frc = jnp.sum(bot_bd_sgm[3])
    return rgt_bd_frc, top_bd_frc, lft_bd_frc, bot_bd_frc

@partial(jit, static_argnums=(4,))
def divergence(Lambda_params, X, bd_X, c2s_params, model):
    X = X[0]
    dsgm_xx_dx = grad_sgm_xx_x(X[:,0], X[:,1], X[:,2], Lambda_params, c2s_params, model)
    dsgm_xy_dy = grad_sgm_xy_y(X[:,0], X[:,1], X[:,2], Lambda_params, c2s_params, model)
    dsgm_yx_dx = grad_sgm_yx_x(X[:,0], X[:,1], X[:,2], Lambda_params, c2s_params, model)
    dsgm_yy_dy = grad_sgm_yy_y(X[:,0], X[:,1], X[:,2], Lambda_params, c2s_params, model)
    
    div_x = jnp.mean(dsgm_xx_dx + dsgm_xy_dy) # = dsgm_xx/dx + dsgm_xy/dy
    div_y = jnp.mean(dsgm_yx_dx + dsgm_yy_dy) # = dsgm_yx/dx + dsgm_yy/dy
    return div_x, div_y

"""
Global training parameters
"""
a1 = 10000.0
a2 = 1.0
lr = 5.e-4 #Original: 5.e-4