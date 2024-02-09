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

from utils_hyperelasticity import eval_P, ThreeDElasticity

import pickle

config.update("jax_enable_x64", True)


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

def train_colloc_parallel(loss, inp, get_params, opt_update, opt_state, key, sharding, fname, nIter = 10000, print_freq=1000, metric_fns=None, batch_size=10000):
    node_X, t_hist = inp
    node_X_ext = jnp.hstack([node_X, t_hist[-1]*np.ones_like(node_X[:,:1])]) #append a row consisting of t_hist[-1]
    val_loss = []
    metrics = []
    for it in range(nIter):
        key, subkey = random.split(key)
        XY_colloc = random.uniform(key, (batch_size,2))
        t_colloc = random.choice(key, t_hist, (batch_size,1))
        XYt_colloc = jnp.hstack([XY_colloc, t_colloc])
        XYt_colloc = jax.device_put(XYt_colloc, sharding)
        # X = [XYt_colloc, random.choice(key, t_hist)]
        opt_state = step_colloc(loss, it, get_params, opt_update, opt_state, XYt_colloc)
        if (it+1)% print_freq == 0:
            params = get_params(opt_state)
            # with open('params/incr/'+fname+'_epoch_{}.npy'.format(it+1), 'wb') as f:
            #     pickle.dump(params, f)
            val_loss_value = loss(params, node_X_ext)
            val_loss.append(val_loss_value)
            if metric_fns is not None:
                m = []
                for metric_fn in metric_fns:
                    m.append(metric_fn(params, node_X_ext))
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

def get_P(X, Y, t, Lambda_params, coord_2_strain_params, model):
    F_xx, F_xy, F_yx, F_yy = coords_2_strain_nn(jnp.array([X,Y,t])[None,:], coord_2_strain_params).flatten()

    # get NODE individual-specific params, phi, from the Lambda NN
    Lambda_inp = jnp.array([X,Y]).reshape([-1,2])
    phi = ff_nn(Lambda_inp, Lambda_params).flatten()
    # Make predictions with this NODE
    # mymodel = model(phi)
    # P = eval_P(F_xx, F_xy, F_yx, F_yy, mymodel)
    ugrad = jnp.array([[F_xx-1.0, F_xy],[F_yx, F_yy-1.0]])
    P = ThreeDElasticity(model).ugrad_2_P(ugrad, phi, 2)
    return P[0,0], P[1,0], P[0,1], P[1,1]
get_P_vmap = vmap(get_P, in_axes=(0,0,0,None,None,None), out_axes=(0,0,0,0))

get_Pxx = lambda X, Y, t, a, b, c: get_P(X, Y, t, a, b, c)[0]
get_Pyx = lambda X, Y, t, a, b, c: get_P(X, Y, t, a, b, c)[1]
get_Pxy = lambda X, Y, t, a, b, c: get_P(X, Y, t, a, b, c)[2]
get_Pyy = lambda X, Y, t, a, b, c: get_P(X, Y, t, a, b, c)[3]

Grad_Pxx_X = vmap(grad(get_Pxx, argnums=0), in_axes=(0,0,0,None,None,None), out_axes=0) # should return dsgm_xx/dx
Grad_Pyx_X = vmap(grad(get_Pyx, argnums=0), in_axes=(0,0,0,None,None,None), out_axes=0)
Grad_Pxy_Y = vmap(grad(get_Pyx, argnums=1), in_axes=(0,0,0,None,None,None), out_axes=0)
Grad_Pyy_Y = vmap(grad(get_Pyy, argnums=1), in_axes=(0,0,0,None,None,None), out_axes=0)

@partial(jit, static_argnums=(4,))
def bd_forces(Lambda_params, t, bd_X, c2s_params, model):
    rgt_bd_X, top_bd_X, lft_bd_X, bot_bd_X = bd_X
    rgt_bd_sgm = get_P_vmap(rgt_bd_X[:,0], rgt_bd_X[:,1], t*jnp.ones_like(rgt_bd_X[:,0]), Lambda_params, c2s_params, model)
    top_bd_sgm = get_P_vmap(top_bd_X[:,0], top_bd_X[:,1], t*jnp.ones_like(top_bd_X[:,0]), Lambda_params, c2s_params, model)
    lft_bd_sgm = get_P_vmap(lft_bd_X[:,0], lft_bd_X[:,1], t*jnp.ones_like(lft_bd_X[:,0]), Lambda_params, c2s_params, model)
    bot_bd_sgm = get_P_vmap(bot_bd_X[:,0], bot_bd_X[:,1], t*jnp.ones_like(bot_bd_X[:,0]), Lambda_params, c2s_params, model)

    rgt_bd_frc = jnp.sum(rgt_bd_sgm[0]) # Assume area of edge = 1.0
    top_bd_frc = jnp.sum(top_bd_sgm[3])
    lft_bd_frc = jnp.sum(lft_bd_sgm[0])
    bot_bd_frc = jnp.sum(bot_bd_sgm[3])
    return rgt_bd_frc, top_bd_frc, lft_bd_frc, bot_bd_frc

@partial(jit, static_argnums=(3,))
def divergence(Lambda_params, XYt, c2s_params, model):
    dPxx_dX = Grad_Pxx_X(XYt[:,0], XYt[:,1], XYt[:,2], Lambda_params, c2s_params, model)
    dPxy_dY = Grad_Pxy_Y(XYt[:,0], XYt[:,1], XYt[:,2], Lambda_params, c2s_params, model)
    dPyx_dX = Grad_Pyx_X(XYt[:,0], XYt[:,1], XYt[:,2], Lambda_params, c2s_params, model)
    dPyy_dY = Grad_Pyy_Y(XYt[:,0], XYt[:,1], XYt[:,2], Lambda_params, c2s_params, model)
    
    Div_X = jnp.mean(dPxx_dX + dPxy_dY)
    Div_Y = jnp.mean(dPyx_dX + dPyy_dY)
    return Div_X, Div_Y

"""
Global training parameters
"""
a1 = 10000.0
a2 = 1.0
lr = 5.e-4 #Original: 5.e-4