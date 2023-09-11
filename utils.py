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

@partial(jit, static_argnums=(0,2,3,))
def step_jp(loss, i, get_params, opt_update, opt_state, X_batch, key):
    params = get_params(opt_state)
    g = grad(loss)(params, X_batch, key)
    return opt_update(i, g, opt_state)

def train_jp(loss, X, get_params, opt_update, opt_state, key, nIter = 10000, print_freq=1000, metric_fns=None, batch_size=None):
    train_loss = []
    metrics = []
    for it in range(nIter):
        key, subkey = random.split(key)
        if batch_size is None:
            X_batch = X
        else:
            X_batch = random.choice(key=key, a=X, shape=(batch_size,), replace=False)
        opt_state = step_jp(loss, it, get_params, opt_update, opt_state, X_batch, key)         
        if (it+1)% print_freq == 0:
            params = get_params(opt_state)
            train_loss_value = loss(params, X, key)
            train_loss.append(train_loss_value)
            if metric_fns is not None:
                m = []
                for metric_fn in metric_fns:
                    m.append(metric_fn(params, X, key))
                metrics.append(m)
            to_print = "it %i, train loss = %e" % (it+1, train_loss_value)
            print(to_print)
    return get_params(opt_state), train_loss, metrics


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
